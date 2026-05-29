import matplotlib
try:
    import psutil
except Exception:
    psutil = None

matplotlib.use('agg')
import numpy as np
import time
import os
import csv
import inspect
from time import perf_counter
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from os.path import join as pjoin
from parser import parameter_parser
from load_data import split_ids, GraphData, collate_batch
from models.hgnn_model import HGNN
from tools.hyperedge_builder import make_hyperedges
from tools.ExpRecorder import ExpRecorder
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score

# Hypergraph cache: key = graph id and hyperedge configuration, value = (H_cpu, De_cpu, Dv_cpu)
hypergraph_cache = {}
# ===================== Profiling Utils =====================
class SimpleProfiler:
    """
    Collect:
      (1) efficiency summary: preprocess / train / inference time + peak mem
      (2) per-graph stats for scalability curves
    """
    def __init__(self, out_dir: str, dataset: str, tag: str, device: str):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        self.dataset = dataset
        self.tag = tag
        self.device = device

        # per-graph rows: one row per (gid, config) when first built + later infer aggregation
        self.graph_rows = {}  # key -> dict

        # epoch rows
        self.epoch_rows = []

        # summary accumulators
        self.prebuild_total_ms = 0.0
        self.prebuild_graphs = 0

        self.train_epoch_ms = []
        self.infer_graph_ms = []  # per graph (measured inside forward)
        self.infer_graphs = 0

        self.gpu_peak_train_mb = 0.0
        self.gpu_peak_infer_mb = 0.0
        self.ram_peak_mb = 0.0

    def _update_ram_peak(self):
        if psutil is None:
            return
        rss = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
        self.ram_peak_mb = max(self.ram_peak_mb, float(rss))

    def _cuda_sync(self):
        if "cuda" in str(self.device) and torch.cuda.is_available():
            torch.cuda.synchronize()

    def log_prebuild(self, ms: float, n_graphs: int = 1):
        self.prebuild_total_ms += float(ms) * int(n_graphs)
        self.prebuild_graphs += int(n_graphs)
        self._update_ram_peak()

    def log_graph_build(self, key, gid: int, n_nodes: int, n_edges: int, n_hyperedges: int, build_ms: float):
        # only keep the first build record (cache miss)
        if key not in self.graph_rows:
            self.graph_rows[key] = {
                "dataset": self.dataset,
                "tag": self.tag,
                "gid": int(gid),
                "n_nodes": int(n_nodes),
                "n_edges": int(n_edges),
                "n_hyperedges": int(n_hyperedges),
                "build_ms": float(build_ms),
                "infer_ms_sum": 0.0,
                "infer_calls": 0,
            }
        self._update_ram_peak()

    def log_graph_infer(self, key, infer_ms: float):
        self.infer_graph_ms.append(float(infer_ms))
        self.infer_graphs += 1
        if key in self.graph_rows:
            self.graph_rows[key]["infer_ms_sum"] += float(infer_ms)
            self.graph_rows[key]["infer_calls"] += 1
        self._update_ram_peak()

    def log_epoch(self, epoch_idx: int, train_ms: float):
        self.train_epoch_ms.append(float(train_ms))
        self.epoch_rows.append({
            "dataset": self.dataset,
            "tag": self.tag,
            "epoch": int(epoch_idx),
            "train_epoch_ms": float(train_ms),
        })
        self._update_ram_peak()

    def update_gpu_peak_train(self):
        if "cuda" in str(self.device) and torch.cuda.is_available():
            mb = torch.cuda.max_memory_allocated() / (1024**2)
            self.gpu_peak_train_mb = max(self.gpu_peak_train_mb, float(mb))

    def update_gpu_peak_infer(self):
        if "cuda" in str(self.device) and torch.cuda.is_available():
            mb = torch.cuda.max_memory_allocated() / (1024**2)
            self.gpu_peak_infer_mb = max(self.gpu_peak_infer_mb, float(mb))

    def dump_csvs(self):
        # 1) efficiency summary
        summary_path = os.path.join(self.out_dir, "efficiency_summary.csv")
        mean_pre_ms = (self.prebuild_total_ms / max(self.prebuild_graphs, 1))
        mean_train_ms = (sum(self.train_epoch_ms) / max(len(self.train_epoch_ms), 1))
        mean_infer_ms = (sum(self.infer_graph_ms) / max(len(self.infer_graph_ms), 1))

        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "dataset", "tag",
                "preprocess_ms_per_graph",
                "train_ms_per_epoch",
                "infer_ms_per_graph",
                "gpu_peak_train_mb",
                "gpu_peak_infer_mb",
                "ram_peak_mb",
                "num_prebuilt_graphs",
                "num_infer_graphs",
            ])
            w.writeheader()
            w.writerow({
                "dataset": self.dataset,
                "tag": self.tag,
                "preprocess_ms_per_graph": f"{mean_pre_ms:.4f}",
                "train_ms_per_epoch": f"{mean_train_ms:.4f}",
                "infer_ms_per_graph": f"{mean_infer_ms:.4f}",
                "gpu_peak_train_mb": f"{self.gpu_peak_train_mb:.2f}",
                "gpu_peak_infer_mb": f"{self.gpu_peak_infer_mb:.2f}",
                "ram_peak_mb": f"{self.ram_peak_mb:.2f}",
                "num_prebuilt_graphs": int(self.prebuild_graphs),
                "num_infer_graphs": int(self.infer_graphs),
            })

        # 2) per-graph for scalability
        per_graph_path = os.path.join(self.out_dir, "scalability_per_graph.csv")
        with open(per_graph_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "dataset", "tag", "gid",
                "n_nodes", "n_edges", "n_hyperedges",
                "build_ms", "infer_ms_avg", "infer_calls",
            ])
            w.writeheader()
            for _, row in self.graph_rows.items():
                calls = max(int(row["infer_calls"]), 1)
                infer_avg = float(row["infer_ms_sum"]) / calls
                w.writerow({
                    "dataset": row["dataset"],
                    "tag": row["tag"],
                    "gid": row["gid"],
                    "n_nodes": row["n_nodes"],
                    "n_edges": row["n_edges"],
                    "n_hyperedges": row["n_hyperedges"],
                    "build_ms": f"{row['build_ms']:.4f}",
                    "infer_ms_avg": f"{infer_avg:.4f}",
                    "infer_calls": int(row["infer_calls"]),
                })

        print(f"[PROFILE] saved: {summary_path}")
        print(f"[PROFILE] saved: {per_graph_path}")


PROF = None  # global profiler (optional)


def _he_config_from_args(args):
    use_struct = not bool(getattr(args, "no_struct_he", False))
    use_coperm = not bool(getattr(args, "no_coperm_he", False))
    use_callctx = not bool(getattr(args, "no_callctx_he", False))
    return use_struct, use_coperm, use_callctx


def _cache_key(gid: int, args):
    use_struct, use_coperm, use_callctx = _he_config_from_args(args)
    return (int(gid), int(use_struct), int(use_coperm), int(use_callctx), int(args.k_call_ctx), int(args.d_struct))


def _build_edges_from_adj(Ab: torch.Tensor):
    # Ab is [n,n] float tensor (cpu)
    src, dst = (Ab > 0).nonzero(as_tuple=True)
    edges = [[int(s), int(d)] for s, d in zip(src.tolist(), dst.tolist())]
    return edges


def _call_make_hyperedges(n, edges, args):
    """
    Compatible call: some versions of make_hyperedges have use_struct/use_coperm/use_callctx,
    some do not. We detect signature dynamically.
    """
    use_struct, use_coperm, use_callctx = _he_config_from_args(args)
    sig = inspect.signature(make_hyperedges)
    kwargs = dict(
        n_nodes=n,
        edges=edges,
        k_call_ctx=args.k_call_ctx,
        d_struct=args.d_struct,
        use_pairwise=True,  # ALWAYS keep pairwise hyperedges
    )
    if "use_struct" in sig.parameters:
        kwargs["use_struct"] = use_struct
    if "use_coperm" in sig.parameters:
        kwargs["use_coperm"] = use_coperm
    if "use_callctx" in sig.parameters:
        kwargs["use_callctx"] = use_callctx
    return make_hyperedges(**kwargs)

print('using torch', torch.__version__)
args = parameter_parser()
# This GitHub version keeps only the HyperSVD/HGNN model.
args.model = 'hgnn'
args.filters = list(map(int, args.filters.split(',')))
args.lr_decay_steps = list(map(int, args.lr_decay_steps.split(',')))
# Provide a default output directory if save_dir is not defined in the parser.
if not hasattr(args, "save_dir"):
    args.save_dir = "./outputs"
for arg in vars(args):
    print(arg, getattr(args, arg))

n_folds = args.folds
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
rnd_state = np.random.RandomState(args.seed)

# Initialize the experiment recorder.
# ---- make sure exp_tag exists ----
if not hasattr(args, "exp_tag") or args.exp_tag is None:
    args.exp_tag = "full"

def _sanitize_tag(s: str) -> str:
    s = str(s)
    for ch in ['/', '\\', ' ', ':', '*', '?', '"', '<', '>', '|']:
        s = s.replace(ch, '_')
    return s

tag = _sanitize_tag(args.exp_tag).strip("_")

# ---- auto tag for ablation flags (append only if not already present) ----
auto = []
if getattr(args, "disable_subattn", False):
    auto.append("woSubAttn")
if getattr(args, "disable_hgconv", False):
    auto.append("woHGConv")

# normalize for contains-check
tag_lower = tag.lower()
auto_final = []
for a in auto:
    if a.lower() not in tag_lower:
        auto_final.append(a)

# build final tag without duplicates
parts = []
if tag and tag != "full":
    parts.append(tag)
parts.extend(auto_final)

final_tag = "_".join(parts) if parts else "full"

rec_dir = os.path.join(
    args.save_dir,
    f"{args.dataset}_{args.model}_{final_tag}_{int(time.time())}"
)
rec = ExpRecorder(out_dir=rec_dir, main_metric="val_f1", mode="max")
meta = {"dataset": args.dataset, "model": args.model, "args": vars(args)}
rec._save_meta(meta)

# ===================== enable profiler =====================
if getattr(args, "profile", False):
    profile_dir = getattr(args, "profile_dir", "")
    if not profile_dir:
        profile_dir = os.path.join(args.save_dir, "profile", args.dataset, final_tag)

    args.profile_dir = profile_dir
    PROF = SimpleProfiler(
        out_dir=profile_dir,
        dataset=args.dataset,
        tag=final_tag,
        device=str(args.device)
    )
    print(f"[PROFILE] enabled, out_dir={profile_dir}")

print('Loading training_data...')


class DataReader():
    """
    Class to read the txt files containing all training_data of the dataset
    """

    def __init__(self, data_dir, rnd_state=None, use_cont_node_attr=False, folds=n_folds):
        self.data_dir = data_dir
        self.rnd_state = np.random.RandomState() if rnd_state is None else rnd_state
        self.use_cont_node_attr = use_cont_node_attr
        files = os.listdir(self.data_dir)
        data = {}
        nodes, graphs, unique_id = self.read_graph_nodes_relations(
            list(filter(lambda f: f.find('graph_indicator') >= 0, files))[0])
        data['features'] = self.read_node_features(list(filter(lambda f: f.find('node_labels') >= 0, files))[0],
                                                   nodes, graphs, fn=lambda s: int(s.strip()))
        data['adj_list'] = self.read_graph_adj(list(filter(lambda f: f.find('_A') >= 0, files))[0], nodes, graphs)
        data['targets'] = np.array(
            self.parse_txt_file(list(filter(lambda f: f.find('graph_labels') >= 0, files))[0],
                                line_parse_fn=lambda s: int(float(s.strip()))))
        data['ids'] = unique_id
        if self.use_cont_node_attr:
            data['attr'] = self.read_node_features(list(filter(lambda f: f.find('node_attributes') >= 0, files))[0],
                                                   nodes, graphs,
                                                   fn=lambda s: np.array(list(map(float, s.strip().split(',')))))
        features, n_edges, degrees = [], [], []
        for sample_id, adj in enumerate(data['adj_list']):
            N = len(adj)  # number of nodes
            if data['features'] is not None:
                assert N == len(data['features'][sample_id]), (N, len(data['features'][sample_id]))
            n = np.sum(adj)  # total sum of edges
            # assert n % 2 == 0, n
            n_edges.append(int(n / 2))  # undirected edges, so need to divide by 2
            if not np.allclose(adj, adj.T):
                print(sample_id, 'not symmetric')
            degrees.extend(list(np.sum(adj, 1)))
            features.append(np.array(data['features'][sample_id]))

        # Create features over graphs as one-hot vectors for each node
        features_all = np.concatenate(features)
        features_min = features_all.min()
        num_features = int(features_all.max() - features_min + 1)  # number of possible values

        features_onehot = []
        for i, x in enumerate(features):
            feature_onehot = np.zeros((len(x), num_features))
            for node, value in enumerate(x):
                feature_onehot[node, value - features_min] = 1
            if self.use_cont_node_attr:
                feature_onehot = np.concatenate((feature_onehot, np.array(data['attr'][i])), axis=1)
            features_onehot.append(feature_onehot)

        if self.use_cont_node_attr:
            num_features = features_onehot[0].shape[1]

        shapes = [len(adj) for adj in data['adj_list']]
        labels = data['targets']  # graph class labels
        labels -= np.min(labels)  # to start from 0

        classes = np.unique(labels)
        num_classes = len(classes)

        if not np.all(np.diff(classes) == 1):
            print('making labels sequential, otherwise pytorch might crash')
            labels_new = np.zeros(labels.shape, dtype=labels.dtype) - 1
            for lbl in range(num_classes):
                labels_new[labels == classes[lbl]] = lbl
            labels = labels_new
            classes = np.unique(labels)
            assert len(np.unique(labels)) == num_classes, np.unique(labels)

        for lbl in classes:
            print('Class %d: \t\t\t%d samples' % (lbl, np.sum(labels == lbl)))

        for u in np.unique(features_all):
            print('feature {}, count {}/{}'.format(u, np.count_nonzero(features_all == u), len(features_all)))

        N_graphs = len(labels)  # number of samples (graphs) in training_data
        assert N_graphs == len(data['adj_list']) == len(features_onehot), 'invalid training_data'

        # Create test sets first
        train_ids, test_ids = split_ids(rnd_state.permutation(N_graphs), folds=folds)

        # Create train sets
        splits = []
        for fold in range(folds):
            splits.append({'train': train_ids[fold], 'test': test_ids[fold]})

        data['features_onehot'] = features_onehot
        data['targets'] = labels
        data['splits'] = splits
        data['N_nodes_max'] = np.max(shapes)  # max number of nodes
        data['num_features'] = num_features
        data['num_classes'] = num_classes
        self.data = data

    def parse_txt_file(self, fpath, line_parse_fn=None):
        with open(pjoin(self.data_dir, fpath), 'r') as f:
            lines = f.readlines()
        data = [line_parse_fn(s) if line_parse_fn is not None else s for s in lines]
        return data

    def read_graph_adj(self, fpath, nodes, graphs):
        edges = self.parse_txt_file(fpath, line_parse_fn=lambda s: s.split(','))
        adj_dict = {}
        for edge in edges:
            node1 = int(edge[0].strip()) - 1  # -1 because of zero-indexing in our code
            node2 = int(edge[1].strip()) - 1
            graph_id = nodes[node1]
            assert graph_id == nodes[node2], ('invalid training_data', graph_id, nodes[node2])
            if graph_id not in adj_dict:
                n = len(graphs[graph_id])
                adj_dict[graph_id] = np.zeros((n, n))
            ind1 = np.where(graphs[graph_id] == node1)[0]
            ind2 = np.where(graphs[graph_id] == node2)[0]
            assert len(ind1) == len(ind2) == 1, (ind1, ind2)
            adj_dict[graph_id][ind1, ind2] = 1
        adj_list = [adj_dict[graph_id] for graph_id in sorted(list(graphs.keys()))]
        return adj_list

    def read_graph_nodes_relations(self, fpath):
        graph_ids = self.parse_txt_file(fpath, line_parse_fn=lambda s: int(s.rstrip()))
        nodes, graphs = {}, {}
        for node_id, graph_id in enumerate(graph_ids):
            if graph_id not in graphs:
                graphs[graph_id] = []
            graphs[graph_id].append(node_id)
            nodes[node_id] = graph_id
        graph_ids = np.unique(list(graphs.keys()))
        unique_id = graph_ids
        for graph_id in graph_ids:
            graphs[graph_id] = np.array(graphs[graph_id])
        return nodes, graphs, unique_id

    def read_node_features(self, fpath, nodes, graphs, fn):
        node_features_all = self.parse_txt_file(fpath, line_parse_fn=fn)
        node_features = {}
        for node_id, x in enumerate(node_features_all):
            graph_id = nodes[node_id]
            if graph_id not in node_features:
                node_features[graph_id] = [None] * len(graphs[graph_id])
            ind = np.where(graphs[graph_id] == node_id)[0]
            assert len(ind) == 1, ind
            assert node_features[graph_id][ind[0]] is None, node_features[graph_id][ind[0]]
            node_features[graph_id][ind[0]] = x
        node_features_lst = [node_features[graph_id] for graph_id in sorted(list(graphs.keys()))]
        return node_features_lst

def hgnn_forward_batch(model, data, device):
    """
    Build hypergraph on cache-miss; record per-graph build/infer stats if PROF enabled.
    Cache key includes ablation flags + (K, d_struct) to avoid collisions.
    """
    x, A, graph_support, N_nodes, labels, ids = data
    B = x.size(0)
    logits = []

    for b in range(B):
        gid = int(ids[b].item())
        n = int(N_nodes[b].item())
        Xb = x[b, :n, :]
        Ab = A[b, :n, :n]

        key = _cache_key(gid, args)

        # -------- build / load hypergraph (CPU cache) --------
        if key in hypergraph_cache:
            H_cpu, De_cpu, Dv_cpu = hypergraph_cache[key]
        else:
            t0 = perf_counter()
            edges = _build_edges_from_adj(Ab.detach().cpu())
            if len(edges) == 0:
                edges = [[i, i] for i in range(n)]

            H_cpu, De_cpu, Dv_cpu = _call_make_hyperedges(n, edges, args)
            hypergraph_cache[key] = (H_cpu, De_cpu, Dv_cpu)

            build_ms = (perf_counter() - t0) * 1000.0
            n_edges = max(len(edges), 0)
            n_hyperedges = int(H_cpu.size(1)) if hasattr(H_cpu, "size") else -1
            if PROF is not None:
                PROF.log_graph_build(key, gid, n, n_edges, n_hyperedges, build_ms)

        # -------- forward (GPU) + per-graph infer time --------
        t1 = perf_counter()
        if "cuda" in str(device) and torch.cuda.is_available():
            torch.cuda.synchronize()

        H = H_cpu.to(device)
        De = De_cpu.to(device)
        Dv = Dv_cpu.to(device)

        logit = model(Xb.to(device), H, De, Dv)
        logits.append(logit.view(1))

        if "cuda" in str(device) and torch.cuda.is_available():
            torch.cuda.synchronize()
        infer_ms = (perf_counter() - t1) * 1000.0
        if PROF is not None:
            PROF.log_graph_infer(key, infer_ms)

    return torch.cat(logits, dim=0)

# ===================== Clean eval-only profiling =====================

def _rss_mb():
    if psutil is None:
        return float("nan")
    return float(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2))


def _gpu_sync(device):
    if "cuda" in str(device) and torch.cuda.is_available():
        torch.cuda.synchronize()


def _gpu_reset_peak(device):
    if "cuda" in str(device) and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _gpu_peak_alloc_mb():
    if torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated() / (1024 ** 2))
    return float("nan")


def _gpu_peak_reserved_mb():
    if torch.cuda.is_available():
        return float(torch.cuda.max_memory_reserved() / (1024 ** 2))
    return float("nan")


def build_clean_eval_cache(loader, args):
    """
    Build hypergraph cache ONLY for eval/test graphs.
    Returns:
        cache: dict[key] = (H_cpu, De_cpu, Dv_cpu)
        rows: per-graph preprocessing rows
    """
    cache = {}
    rows = []

    for batch_idx, data in enumerate(loader):
        x_b, A_b, gs_b, N_b, y_b, ids_b = data
        B = x_b.size(0)

        for b in range(B):
            gid = int(ids_b[b].item())
            n = int(N_b[b].item())
            Ab = A_b[b, :n, :n]

            key = _cache_key(gid, args)
            if key in cache:
                continue

            edges = _build_edges_from_adj(Ab)
            if len(edges) == 0:
                edges = [[i, i] for i in range(n)]

            t0 = perf_counter()
            H_cpu, De_cpu, Dv_cpu = _call_make_hyperedges(n, edges, args)
            _gpu_sync(args.device)
            build_ms = (perf_counter() - t0) * 1000.0

            cache[key] = (H_cpu, De_cpu, Dv_cpu)

            rows.append({
                "gid": gid,
                "n_nodes": n,
                "n_edges": len(edges),
                "n_hyperedges": int(H_cpu.size(1)),
                "preprocess_ms": float(build_ms),
            })

    return cache, rows


@torch.no_grad()
def hgnn_forward_batch_clean(model, data, device, cache, args):
    """
    Eval-only forward using a CLEAN cache built from test graphs only.
    Returns:
        logits: [B]
        rows: per-graph inference rows with per-graph GPU/RAM peaks
    """
    x, A, graph_support, N_nodes, labels, ids = data
    B = x.size(0)
    logits = []
    rows = []

    for b in range(B):
        gid = int(ids[b].item())
        n = int(N_nodes[b].item())
        Xb = x[b, :n, :]
        key = _cache_key(gid, args)
        H_cpu, De_cpu, Dv_cpu = cache[key]

        ram_before = _rss_mb()
        _gpu_reset_peak(device)
        _gpu_sync(device)
        t0 = perf_counter()

        H = H_cpu.to(device)
        De = De_cpu.to(device)
        Dv = Dv_cpu.to(device)
        logit = model(Xb.to(device), H, De, Dv)
        logits.append(logit.view(1))

        _gpu_sync(device)
        infer_ms = (perf_counter() - t0) * 1000.0
        ram_after = _rss_mb()

        rows.append({
            "gid": gid,
            "infer_ms": float(infer_ms),
            "infer_gpu_peak_alloc_mb": _gpu_peak_alloc_mb(),
            "infer_gpu_peak_reserved_mb": _gpu_peak_reserved_mb(),
            "infer_ram_peak_mb": float(np.nanmax([ram_before, ram_after])),
        })

    return torch.cat(logits, dim=0), rows


def run_clean_eval_profile(model, loader, args, fold_id):
    """
    Clean eval-only profiling for ONE fold:
      1) build eval cache on test graphs only
      2) run one no-grad eval pass
      3) output clean summary + per-graph rows
    """
    model.eval()

    # -------- step1: preprocessing on test set only --------
    if "cuda" in str(args.device) and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    ram_peak = _rss_mb()
    t0 = perf_counter()
    clean_cache, pre_rows = build_clean_eval_cache(loader, args)
    _gpu_sync(args.device)
    _ = (perf_counter() - t0) * 1000.0  # total preprocess wall time (not used below)
    ram_peak = max(ram_peak, _rss_mb())

    # gid -> row
    row_map = {}
    for r in pre_rows:
        gid = int(r["gid"])
        row_map[gid] = {
            "dataset": args.dataset,
            "tag": final_tag if 'final_tag' in globals() else str(getattr(args, "exp_tag", "default")),
            "fold": int(fold_id),
            "gid": gid,
            "n_nodes": int(r["n_nodes"]),
            "n_edges": int(r["n_edges"]),
            "n_hyperedges": int(r["n_hyperedges"]),
            "preprocess_ms": float(r["preprocess_ms"]),
            "infer_ms": 0.0,
            "end2end_ms": 0.0,
            "infer_gpu_peak_alloc_mb": float("nan"),
            "infer_gpu_peak_reserved_mb": float("nan"),
            "infer_ram_peak_mb": float("nan"),
        }

    preprocess_ms_list = [r["preprocess_ms"] for r in pre_rows]

    # -------- step2: eval-only inference --------
    if "cuda" in str(args.device) and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    infer_ms_list = []

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            for i in range(len(data)):
                data[i] = data[i].to(args.device)

            logits, infer_rows = hgnn_forward_batch_clean(
                model=model,
                data=data,
                device=args.device,
                cache=clean_cache,
                args=args
            )

            for rr in infer_rows:
                gid = int(rr["gid"])
                infer_ms = float(rr["infer_ms"])
                infer_ms_list.append(infer_ms)
                if gid in row_map:
                    row_map[gid]["infer_ms"] = infer_ms
                    row_map[gid]["end2end_ms"] = row_map[gid]["preprocess_ms"] + infer_ms
                    row_map[gid]["infer_gpu_peak_alloc_mb"] = float(rr["infer_gpu_peak_alloc_mb"])
                    row_map[gid]["infer_gpu_peak_reserved_mb"] = float(rr["infer_gpu_peak_reserved_mb"])
                    row_map[gid]["infer_ram_peak_mb"] = float(rr["infer_ram_peak_mb"])

            ram_peak = max(ram_peak, _rss_mb())

    gpu_alloc_mb = _gpu_peak_alloc_mb()
    gpu_reserved_mb = _gpu_peak_reserved_mb()

    per_graph_rows = list(row_map.values())

    summary_row = {
        "dataset": args.dataset,
        "tag": final_tag if 'final_tag' in globals() else str(getattr(args, "exp_tag", "default")),
        "fold": int(fold_id),
        "num_eval_graphs": len(per_graph_rows),
        "preprocess_ms_per_graph": float(np.mean(preprocess_ms_list)) if len(preprocess_ms_list) else float("nan"),
        "infer_ms_per_graph": float(np.mean(infer_ms_list)) if len(infer_ms_list) else float("nan"),
        "end2end_ms_per_graph": float(np.mean([r["end2end_ms"] for r in per_graph_rows])) if len(per_graph_rows) else float("nan"),
        "throughput_graphs_per_sec": float(1000.0 / np.mean([r["end2end_ms"] for r in per_graph_rows])) if len(per_graph_rows) else float("nan"),
        "gpu_peak_alloc_mb": gpu_alloc_mb,
        "gpu_peak_reserved_mb": gpu_reserved_mb,
        "ram_peak_mb": ram_peak,
        "avg_infer_gpu_peak_alloc_mb": float(np.nanmean([r["infer_gpu_peak_alloc_mb"] for r in per_graph_rows])) if len(per_graph_rows) else float("nan"),
        "avg_infer_gpu_peak_reserved_mb": float(np.nanmean([r["infer_gpu_peak_reserved_mb"] for r in per_graph_rows])) if len(per_graph_rows) else float("nan"),
        "avg_infer_ram_peak_mb": float(np.nanmean([r["infer_ram_peak_mb"] for r in per_graph_rows])) if len(per_graph_rows) else float("nan"),
        "avg_nodes": float(np.mean([r["n_nodes"] for r in per_graph_rows])) if len(per_graph_rows) else float("nan"),
        "avg_edges": float(np.mean([r["n_edges"] for r in per_graph_rows])) if len(per_graph_rows) else float("nan"),
        "avg_hyperedges": float(np.mean([r["n_hyperedges"] for r in per_graph_rows])) if len(per_graph_rows) else float("nan"),
    }

    return summary_row, per_graph_rows


def dump_clean_eval_csv(profile_dir, summary_rows, per_graph_rows):
    os.makedirs(profile_dir, exist_ok=True)

    summary_csv = os.path.join(profile_dir, "BlockInfoDep_clean_eval_summary.csv")
    graph_csv = os.path.join(profile_dir, "clean_eval_per_graph.csv")

    # ---------- summary ----------
    if len(summary_rows) > 0:
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)

            # avg row over folds
            avg_row = {"dataset": summary_rows[0]["dataset"], "tag": summary_rows[0]["tag"], "fold": "avg"}
            num_keys = [
                "num_eval_graphs",
                "train_ms_per_epoch",
                "preprocess_ms_per_graph",
                "infer_ms_per_graph",
                "end2end_ms_per_graph",
                "throughput_graphs_per_sec",
                "gpu_peak_alloc_mb",
                "gpu_peak_reserved_mb",
                "ram_peak_mb",
                "avg_infer_gpu_peak_alloc_mb",
                "avg_infer_gpu_peak_reserved_mb",
                "avg_infer_ram_peak_mb",
                "avg_nodes",
                "avg_edges",
                "avg_hyperedges",
            ]
            for k in num_keys:
                vals = [float(r[k]) for r in summary_rows]
                avg_row[k] = float(np.mean(vals))
            writer.writerow(avg_row)

    # ---------- per graph ----------
    if len(per_graph_rows) > 0:
        with open(graph_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_graph_rows[0].keys()))
            writer.writeheader()
            for row in per_graph_rows:
                writer.writerow(row)

    print(f"[CLEAN-PROFILE] saved: {summary_csv}")
    print(f"[CLEAN-PROFILE] saved: {graph_csv}")

datareader = DataReader(data_dir='./training_data/%s/' % args.dataset, rnd_state=rnd_state,
                        use_cont_node_attr=args.use_cont_node_attr, folds=args.folds)
cv_true = []
cv_score = []

# Train and test.
result_folds = []
clean_eval_summary_rows = []
clean_eval_per_graph_rows = []
for fold_id in range(n_folds):
    loaders = []
    for split in ['train', 'test']:
        gdata = GraphData(fold_id=fold_id, datareader=datareader, split=split)
        loader = DataLoader(
            gdata,
            batch_size=args.batch_size,
            shuffle=split.find('train') >= 0,
            num_workers=args.threads,
            collate_fn=collate_batch
        )
        loaders.append(loader)

    print('FOLD {}, train {}, test {}'.format(
        fold_id, len(loaders[0].dataset), len(loaders[1].dataset))
    )

    # ===================== build HGNN model =====================
    model = HGNN(
        in_dim=loaders[0].dataset.num_features,
        hid_dim=args.n_hidden,
        n_layers=len(args.filters),
        alpha_hg=args.alpha_hg,
        k_call_ctx=args.k_call_ctx,
        d_struct=args.d_struct,
        dropout_p=args.dropout,
        disable_subattn=args.disable_subattn,
        disable_hgconv=args.disable_hgconv,
    ).to(args.device)

    # ===================== offline preprocessing (hypergraph prebuild) =====================
    if PROF is not None and (not getattr(args, "no_prebuild_cache", False)):
        t0 = perf_counter()
        before = len(hypergraph_cache)

        for _loader in loaders:
            for batch_idx, data in enumerate(_loader):
                x_b, A_b, gs_b, N_b, y_b, ids_b = data
                B = x_b.size(0)

                for b in range(B):
                    gid = int(ids_b[b].item())
                    n = int(N_b[b].item())
                    Ab = A_b[b, :n, :n]

                    key = _cache_key(gid, args)
                    if key in hypergraph_cache:
                        continue

                    edges = _build_edges_from_adj(Ab)
                    if len(edges) == 0:
                        edges = [[i, i] for i in range(n)]

                    H_cpu, De_cpu, Dv_cpu = _call_make_hyperedges(n, edges, args)
                    hypergraph_cache[key] = (H_cpu, De_cpu, Dv_cpu)

        if "cuda" in str(args.device) and torch.cuda.is_available():
            torch.cuda.synchronize()

        ms = (perf_counter() - t0) * 1000.0
        added = len(hypergraph_cache) - before

        if added > 0:
            avg_ms = ms / added
            PROF.log_prebuild(avg_ms, n_graphs=added)

        print(
            f"[PROFILE] prebuild_cache: added_graphs={added}, "
            f"total_ms={ms:.2f}, avg_ms_per_graph={ms / max(added, 1):.4f}"
        )

    print('Initialize model...')

    train_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    print('N trainable parameters:', np.sum([p.numel() for p in train_params]))
    optimizer = optim.Adam(train_params, lr=args.lr, betas=(0.5, 0.999), weight_decay=args.wd)
    scheduler = lr_scheduler.MultiStepLR(optimizer, args.lr_decay_steps, gamma=0.1)
    loss_fn_train = nn.BCEWithLogitsLoss()
    loss_fn_test = nn.BCEWithLogitsLoss(reduction='sum')


    def train(train_loader):
        t_epoch0 = perf_counter()
        if PROF is not None and "cuda" in str(args.device) and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        model.train()
        start = time.time()
        train_loss, n_samples = 0, 0
        for batch_idx, data in enumerate(train_loader):
            for i in range(len(data)):
                data[i] = data[i].to(args.device)
            optimizer.zero_grad()
            output = hgnn_forward_batch(model, data, args.device)
            loss = loss_fn_train(output, data[4].float())
            batch_size = len(output)

            loss.backward()
            optimizer.step()
            time_iter = time.time() - start
            train_loss += loss.item() * batch_size
            n_samples += batch_size
        avg_train_loss = train_loss / n_samples
        print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} (avg: {:.6f})  sec/iter: {:.4f}'.format(
            epoch + 1, n_samples, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader),
            loss.item(), avg_train_loss, time_iter / (batch_idx + 1)))
        if "cuda" in str(args.device) and torch.cuda.is_available():
            torch.cuda.synchronize()
        train_ms = (perf_counter() - t_epoch0) * 1000.0
        if PROF is not None:
            PROF.log_epoch(epoch_idx=epoch, train_ms=train_ms)
            PROF.update_gpu_peak_train()
        return avg_train_loss


    def test(test_loader):
        if PROF is not None and "cuda" in str(args.device) and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        model.eval()
        start = time.time()
        test_loss, n_samples, count = 0, 0, 0
        tn, fp, fn, tp = 0, 0, 0, 0
        accuracy, recall, precision, F1 = 0, 0, 0, 0
        fn_list = []
        fp_list = []
        all_true = []
        all_score = []

        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader):
                for i in range(len(data)):
                    data[i] = data[i].to(args.device)

                output = hgnn_forward_batch(model, data, args.device)
                loss = loss_fn_test(output, data[4].float())
                prob = torch.sigmoid(output).detach().cpu()
                test_loss += loss.item()
                n_samples += len(output)
                count += 1
                pred = (prob >= 0.5).long().view(-1, 1)
                all_true.append(data[4].detach().cpu().view(-1).numpy())
                all_score.append(prob.detach().cpu().view(-1).numpy())

                for k in range(len(pred)):
                    if (np.array(pred.view_as(data[4])[k]).tolist() == 1) & (
                            np.array(data[4].detach().cpu()[k]).tolist() == 1):
                        tp += 1
                    elif (np.array(pred.view_as(data[4])[k]).tolist() == 0) & (
                            np.array(data[4].detach().cpu()[k]).tolist() == 0):
                        tn += 1
                    elif (np.array(pred.view_as(data[4])[k]).tolist() == 0) & (
                            np.array(data[4].detach().cpu()[k]).tolist() == 1):
                        fn += 1
                        fn_list.append(np.array(data[5].detach().cpu()[k]).tolist())
                    elif (np.array(pred.view_as(data[4])[k]).tolist() == 1) & (
                            np.array(data[4].detach().cpu()[k]).tolist() == 0):
                        fp += 1
                        fp_list.append(np.array(data[5].detach().cpu()[k]).tolist())

                target = data[4].view(-1).cpu().numpy()
                predictions = pred.view_as(data[4]).view(-1).cpu().numpy()
                accuracy += metrics.accuracy_score(target, predictions)
                recall += metrics.recall_score(target, predictions, zero_division=0)
                precision += metrics.precision_score(target, predictions, zero_division=0)
                F1 += metrics.f1_score(target, predictions, zero_division=0)

        print(tp, fp, tn, fn)
        accuracy = 100. * accuracy / count
        recall = 100. * recall / count
        precision = 100. * precision / count
        F1 = 100. * F1 / count
        FPR = fp / (fp + tn)
        avg_test_loss = test_loss / n_samples

        print(
            'Test set (epoch {}): Average loss: {:.4f}, Accuracy: ({:.2f}%), Recall: ({:.2f}%), Precision: ({:.2f}%), '
            'F1-Score: ({:.2f}%), FPR: ({:.2f}%)  sec/iter: {:.4f}\n'.format(
                epoch + 1, avg_test_loss, accuracy, recall, precision, F1, FPR,
                (time.time() - start) / len(test_loader))
        )

        print("fn_list(predict == 0 & label == 1):", fn_list)
        print("fp_list(predict == 1 & label == 0):", fp_list)
        print()

        y_true = np.concatenate(all_true, axis=0) if len(all_true) else np.array([])
        y_score = np.concatenate(all_score, axis=0) if len(all_score) else np.array([])
        if y_true.size > 0:
            fpr_arr, tpr_arr, _ = roc_curve(y_true, y_score)
            auc_val = roc_auc_score(y_true, y_score)
        else:
            fpr_arr, tpr_arr, auc_val = np.array([]), np.array([]), float("nan")

        print(f"[ROC] AUC = {auc_val:.4f}")
        if PROF is not None:
            PROF.update_gpu_peak_infer()

        return avg_test_loss, accuracy, recall, precision, F1, FPR, auc_val, fpr_arr, tpr_arr, y_true, y_score

    best_f1 = -1.0
    best_pack = None  # (val_loss, acc, recall, precision, f1, fpr, auc, y_true, y_score)
    best_state = None

    for epoch in range(args.epochs):
        train_loss = train(loaders[0])
        val_loss, accuracy, recall, precision, F1, FPR, auc_val, fpr_arr, tpr_arr, y_true, y_score = test(loaders[1])
        scheduler.step()
        metric = {
            "fold": fold_id,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_acc": float(accuracy),
            "val_recall": float(recall),
            "val_precision": float(precision),
            "val_f1": float(F1),
            "val_fpr": float(FPR),
            "val_auc": float(auc_val),
        }

        global_epoch = fold_id * args.epochs + epoch
        meta_fold = dict(meta)
        meta_fold["fold"] = fold_id

        rec.save_epoch(
            epoch=global_epoch,
            metrics=metric,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            meta=meta_fold,
            save_ckpt_always=False,
        )

        # Keep the best checkpoint according to validation F1.
        if F1 > best_f1:
            best_f1 = F1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_pack = (val_loss, accuracy, recall, precision, F1, FPR, auc_val, y_true, y_score)

    # Restore the best model and use its metrics for this fold.
    if best_state is not None:
        model.load_state_dict(best_state)

    val_loss, accuracy, recall, precision, F1, FPR, auc_val, y_true, y_score = best_pack
    cv_true.append(y_true)
    cv_score.append(y_score)
    result_folds.append([accuracy, recall, precision, F1, FPR])
    # ===================== clean eval-only profiling (one pass on test set) =====================
    if getattr(args, "profile", False):
        clean_summary_row, clean_graph_rows = run_clean_eval_profile(
            model=model,
            loader=loaders[1],   # test loader only
            args=args,
            fold_id=fold_id
        )
        if PROF is not None and len(PROF.train_epoch_ms) >= args.epochs:
            fold_train_ms = float(np.mean(PROF.train_epoch_ms[-args.epochs:]))
        else:
            fold_train_ms = float("nan")

        clean_summary_row["train_ms_per_epoch"] = fold_train_ms
        clean_eval_summary_rows.append(clean_summary_row)
        clean_eval_per_graph_rows.extend(clean_graph_rows)

        print(
            f"[CLEAN-PROFILE][fold={fold_id}] "
            f"pre={clean_summary_row['preprocess_ms_per_graph']:.4f} ms/graph, "
            f"infer={clean_summary_row['infer_ms_per_graph']:.4f} ms/graph, "
            f"end2end={clean_summary_row['end2end_ms_per_graph']:.4f} ms/graph, "
            f"throughput={clean_summary_row['throughput_graphs_per_sec']:.2f} graphs/s"
        )

# ===================== dump clean eval-only profiling CSVs =====================
if getattr(args, "profile", False) and len(clean_eval_summary_rows) > 0:
    profile_dir = getattr(args, "profile_dir", "")
    if not profile_dir:
        profile_dir = os.path.join(args.save_dir, "profile", args.dataset, final_tag if 'final_tag' in globals() else "default")
    dump_clean_eval_csv(profile_dir, clean_eval_summary_rows, clean_eval_per_graph_rows)

print(result_folds)
acc_list = []
recall_list = []
precision_list = []
F1_list = []
FPR_list = []

for i in range(len(result_folds)):
    acc_list.append(result_folds[i][0])
    recall_list.append(result_folds[i][1])
    precision_list.append(result_folds[i][2])
    F1_list.append(result_folds[i][3])
    FPR_list.append(result_folds[i][4])

print(
    '{}-fold cross validation avg acc (+- std): {}% ({}%), recall (+- std): {}% ({}%), precision (+- std): {}% ({}%), '
    'F1-Score (+- std): {}% ({}%), FPR (+- fpr): {}% ({}%)'.format(
        n_folds, np.mean(acc_list), np.std(acc_list), np.mean(recall_list), np.std(recall_list),
        np.mean(precision_list), np.std(precision_list), np.mean(F1_list), np.std(F1_list), np.mean(FPR_list),
        np.std(FPR_list))
)
# ===== save ROC npz (optional) =====
save_npz = getattr(args, "save_npz", None)

# normalize
if save_npz is None:
    pass
else:
    if isinstance(save_npz, str):
        s = save_npz.strip().lower()
        if s in ["none", "null", "0", "false", ""]:
            save_npz = None
        elif s == "auto":
            # auto path: <save_dir>/roc/<dataset>/<tag>.npz
            # final_tag has been computed above.
            save_npz = os.path.join(args.save_dir, "roc", args.dataset, f"{final_tag}.npz")

if save_npz is not None:
    dirn = os.path.dirname(save_npz)
    if dirn:  # Avoid calling makedirs when dirname is empty.
        os.makedirs(dirn, exist_ok=True)

    y_true_all = np.concatenate(cv_true, axis=0) if len(cv_true) else np.array([])
    y_score_all = np.concatenate(cv_score, axis=0) if len(cv_score) else np.array([])

    if y_true_all.size > 0:
        fpr_all, tpr_all, _ = roc_curve(y_true_all, y_score_all)
        auc_all = roc_auc_score(y_true_all, y_score_all)
    else:
        fpr_all, tpr_all, auc_all = np.array([]), np.array([]), float("nan")

    np.savez(
        save_npz,
        y_true=y_true_all,
        y_score=y_score_all,
        fpr=fpr_all,
        tpr=tpr_all,
        auc=auc_all,
        dataset=args.dataset,
        model=args.model,
        exp_tag=args.exp_tag,
        disable_subattn=bool(getattr(args, "disable_subattn", False)),
        disable_hgconv=bool(getattr(args, "disable_hgconv", False)),
        no_callctx_he=bool(getattr(args, "no_callctx_he", False)),
        no_coperm_he=bool(getattr(args, "no_coperm_he", False)),
        no_struct_he=bool(getattr(args, "no_struct_he", False)),
        tag=final_tag,
    )
    print(f"[SAVE] ROC npz saved to: {save_npz}  (AUC={auc_all:.4f})")
else:
    print("[SAVE] save_npz disabled.")
# ===================== dump profiling CSVs =====================
if PROF is not None:
    PROF.dump_csvs()
