# models/hgnn_model.py
import torch
import torch.nn as nn

# ----------------- Hypergraph convolution -----------------
class HypergraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
    def forward(self, X, H, De, Dv):
        # X: N x F ; H: N x E ; De: E ; Dv: N
        De_inv = 1.0 / (De + 1e-9)
        Dv_inv = 1.0 / (Dv + 1e-9)
        H_De = H * De_inv.unsqueeze(0)     # N x E
        Ht = H.t()                          # E x N
        A = torch.matmul(H_De, Ht)          # N x N
        A = A * Dv_inv.unsqueeze(1)
        XW = self.fc(X)
        out = torch.matmul(A, XW)
        return out

# ----------------- Subgraph/hyperedge attention branch -----------------
class SubgraphAttention(nn.Module):
    """
    Treat each hyperedge as a subgraph. The module first obtains a hyperedge representation
    by pooling the nodes contained in each hyperedge, then computes global attention over all
    hyperedges and writes the attended hyperedge messages back to nodes through H. This is a
    stable implementation of subgraph-level attention: it does not assign personalized weights
    to each node, but highlights important hyperedges at the global level.
    """
    def __init__(self, in_dim, hid_dim):
        super().__init__()
        self.edge_proj = nn.Linear(in_dim, hid_dim)
        self.attn_vec  = nn.Linear(hid_dim, 1)

    def forward(self, X, H, De):
        # Hyperedge representation: E_repr = (H^T X) / De
        E_repr = torch.matmul(H.t(), X) / De.unsqueeze(1)  # E x F

        # Project hyperedge representations to the hidden dimension.
        E_emb = self.edge_proj(E_repr)  # E x H
        Z = torch.tanh(E_emb)  # E x H

        # Compute attention scores over all hyperedges.
        scores = self.attn_vec(Z).squeeze(-1)  # E
        alpha_hg = torch.sigmoid(scores)     # E x 1

        # Scale stabilization: keep the mean of alpha_hg close to 1, independent of the number of hyperedges.
        alpha_hg = alpha_hg / (alpha_hg.mean() + 1e-9)

        # Write messages back to nodes.
        E_msg = alpha_hg.unsqueeze(1) * E_emb  # E x hid
        X_sub = torch.matmul(H, E_msg)  # N x H
        # Node-side normalization: divide by the number of incident hyperedges to avoid scale inflation.

        dv = H.sum(dim=1, keepdim=True).clamp(min=1.0)  # N x 1
        X_sub = X_sub / dv

        return X_sub, alpha_hg

class HGNN(nn.Module):
    def __init__(self, in_dim, hid_dim=128, n_layers=2,alpha_hg=0.2, k_call_ctx=1, d_struct=1, dropout_p=0.1, disable_subattn=False, disable_hgconv=False,):
        super().__init__()
        # Store input hyperparameters.
        self.alpha_hg = alpha_hg  # Weight controlling the balance between subgraph attention and hypergraph convolution.
        self.k_call_ctx = k_call_ctx
        self.d_struct = d_struct

        # --- Store key hyperparameters for the experiment recorder. ---
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(dropout_p)
        self.disable_subattn = disable_subattn
        self.disable_hgconv = disable_hgconv
        if self.disable_subattn and self.disable_hgconv:
            raise ValueError("Both branches disabled: disable_subattn=True and disable_hgconv=True")
        # Hypergraph convolution stack.
        self.layers = nn.ModuleList()
        if not self.disable_hgconv:
            self.layers.append(HypergraphConv(in_dim, hid_dim))
            for _ in range(n_layers - 1):
                self.layers.append(HypergraphConv(hid_dim, hid_dim))
        # Subgraph attention branch.
        self.subattn = None
        if not self.disable_subattn:
            self.subattn = SubgraphAttention(in_dim, hid_dim)

        self.ln_h = nn.LayerNorm(hid_dim)
        self.ln_s = nn.LayerNorm(hid_dim)

        # learnable fusion gate: start from alpha_hg, then learn
        init = float(alpha_hg)
        init = min(max(init, 1e-4), 1 - 1e-4)
        self.gate_logit = nn.Parameter(torch.log(torch.tensor(init / (1.0 - init))))

        # Readout after branch fusion.
        self.readout = nn.Sequential(
            nn.Linear(hid_dim, hid_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout_p),  # Dropout in the readout layer.
            nn.Linear(hid_dim//2, 1)
        )

    def forward(self, X, H, De, Dv):
        # Branch 1: hypergraph convolution.
        h = None
        if not self.disable_hgconv:
            h = X
            for l in self.layers:
                h = torch.relu(l(h, H, De, Dv))          # N x H
                h = self.dropout(h)  # Inter-layer dropout.
        # Branch 2: subgraph attention, which directly outputs hidden-dimensional messages.
        sub_msg = None
        if not self.disable_subattn:
            sub_msg, _ = self.subattn(X, H, De)  # N x hid_dim

        # Message fusion: Z = (1 - alpha_hg) * h + alpha_hg * sub_msg.
        # Scale alignment and regularization.
        if h is not None:
            h = self.ln_h(h)
        if sub_msg is not None:
            sub_msg = self.ln_s(sub_msg)
            sub_msg = self.dropout(sub_msg)

        if self.disable_hgconv:
            z = sub_msg
        elif self.disable_subattn:
            z = h
        else:
            alpha_hg = torch.sigmoid(self.gate_logit)  # learnable in (0,1)
            z = (1.0 - alpha_hg) * h + alpha_hg * sub_msg

        # Graph-level readout and classification.
        g = z.mean(dim=0, keepdim=True)              # 1 x hid
        logits = self.readout(g).squeeze(0)          # shape: [1]
        return logits
