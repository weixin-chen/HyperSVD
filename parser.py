import argparse


def parameter_parser():
    # Experiment parameters
    parser = argparse.ArgumentParser(description='Smart contract vulnerability detection based on graph neural network')
    parser.add_argument('-D', '--dataset', type=str, default='Reentrancy',
                        choices=['Reentrancy','NestedCall','BlockInfoDep','TranStaDep'])
    parser.add_argument('-M', '--model', type=str, default='hgnn',
                        choices=['gcn_modify', 'gat', 'gcn_origin','hgnn','tmp'])
    parser.add_argument('--lr', type=float, default=0.002, help='learning rate')
    # Epoch indices for learning-rate decay. For example, '1,3' decays the learning rate at epochs 1 and 3.
    parser.add_argument('--lr_decay_steps', type=str, default='1,3', help='learning rate')
    # Weight decay adds an L2 penalty to keep model weights small and reduce overfitting.
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    # Dropout rate used to randomly drop hidden units during training.
    parser.add_argument('-d', '--dropout', type=float, default=0.2, help='dropout rate')
    # Number of filters in each convolution layer.
    parser.add_argument('-f', '--filters', type=str, default='64,64,64', help='number of filters in each layer')
    # Number of hidden units in the fully connected layer after the last convolution layer.
    parser.add_argument('--n_hidden', type=int, default=64,
                        help='number of hidden units in a fully connected layer after the last conv layer')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    # Batch size, i.e., the number of samples processed in each mini-batch.
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('-t', '--threads', type=int, default=0, help='number of threads to load training_data')
    parser.add_argument('--log_interval', type=int, default=1,
                        help='interval (number of batches) of logging')
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'])
    parser.add_argument('--seed', type=int, default=50, help='random seed')
    parser.add_argument('--shuffle_nodes', action='store_true', default=True, help='shuffle nodes for debugging')
    # Number of folds for cross-validation. For example, 5-fold evaluation uses each split once as the test set and averages the results.
    parser.add_argument('-F', '--folds', default=5, choices=[3, 5, 10], help='n-fold cross validation')
    parser.add_argument('-a', '--adj_sq', action='store_true', default=True,
                        help='use A^2 instead of A as an adjacency matrix')
    parser.add_argument('-s', '--scale_identity', action='store_true', default=False,
                        help='use 2I instead of I for self connections')
    # Whether to use continuous node attributes in addition to discrete node labels.
    parser.add_argument('-c', '--use_cont_node_attr', action='store_true', default=True,
                        help='use continuous node attributes in addition to discrete ones')
    # Negative-slope coefficient for the LeakyReLU activation.
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha value for the leaky_relu')
    parser.add_argument('--multi_head', type=int, default=4, help='number of head attentions(Multi-Head)')
    parser.add_argument('--alpha_hg', type=float, default=0.5,
                        help='Weight for the combination of subgraph attention and hypergraph convolution')
    parser.add_argument('--k_call_ctx', type=int, default=3, help='Context size (k) for call edges')
    parser.add_argument('--d_struct', type=int, default=3, help='Degree for structural edges')
    # ===== Ablation: HGNN branches =====
    parser.add_argument('--disable_subattn', action='store_true', default=False,
                        help='Ablation: remove SubgraphAttention branch')
    parser.add_argument('--disable_hgconv', action='store_true', default=False,
                        help='Ablation: remove HypergraphConv branch True')
    # -------- Ablation: hyperedge families --------
    parser.add_argument("--no_callctx_he", action="store_true", default=False,
                        help="ablation: remove call-context hyperedges (keep others)")
    parser.add_argument("--no_coperm_he", action="store_true", default=False,
                        help="ablation: remove co-permission hyperedges (keep others)")
    parser.add_argument("--no_struct_he", action="store_true", default=False,
                        help="ablation: remove structural hyperedges (keep others)")

    # ===== ROC exporting =====
    parser.add_argument('--exp_tag', type=str,default="d3k3",
                        help='Short tag for this run (used in output naming)')

    # auto: save to outputs/roc/<dataset>/<final_tag>.npz
    parser.add_argument('--save_npz', type=str, default="auto",
                        help='Set path to save ROC npz; use "auto" to save to outputs/roc/<dataset>/<tag>.npz; set "none" to disable')
    # ===== Profiling (efficiency & scalability) =====
    parser.add_argument("--profile", action="store_true",
                        help="Enable efficiency/scalability profiling (write CSVs).")
    parser.add_argument("--profile_dir", type=str, default="",
                        help="Directory to save profiling CSVs (default: <save_dir>/profile).")
    parser.add_argument("--no_prebuild_cache", action="store_true",
                        help="Disable hypergraph prebuild (offline preprocessing) before training.")

    return parser.parse_args()
