import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="scRNA-seq data imputation using CoImpute.")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=int,
                        default=0,
                        help="Specify GPU ID to use (default: 0). Ignored if no CUDA available.")
    parser.add_argument('--graph_AE_cell_batch_size', type=int,
                        default=1024,
                        help='(int, default 1024) Cell Batch size')
    parser.add_argument('--graph_AE_gene_batch_size', type=int,
                        default=1024,
                        help='(int, default 1024) Gene Batch size')
    parser.add_argument('--sc_data', type=str,
                        # default="./data/X.csv",
                        help='Raw single-cell count matrices in formats such as CSV, h5ad, 10x mtx, or TXT were imported into AnnData objects.')
    parser.add_argument('--csv_to_load',
                        # action='store_false', default=True,
                        action='store_true', default=False,
                        help='Whether input is raw count data in CSV format')
    parser.add_argument('--h5ad_to_load',
                        action='store_true', default=False,
                        help='Whether input is raw count data in h5ad format')
    parser.add_argument('--x10_to_load',
                        action='store_true', default=False,
                        help='Whether input is raw count data in 10X format')
    parser.add_argument('--txt_to_load',
                        action='store_true', default=False,
                        help='Whether input is raw count data in TXT format')
    parser.add_argument('-t', '--transpose', dest='transpose',
                        action='store_true', default=False,
                        help='Transpose input matrix (default: False)')
    parser.add_argument('--check_counts',
                        action='store_false', default=True,
                        help='Whether check_counts'
                        )
    parser.add_argument('--filter_min_counts',
                        action='store_true', default=False,
                        help='Whether store_false'
                        )
    parser.add_argument('--min_gene_counts', type=int,
                        default=100,
                        help='(int, default 100) min_gene_counts')
    parser.add_argument('--min_cell_counts', type=int,
                        default=1,
                        help='(int, default 1) min_cell_counts')
    parser.add_argument('--logtrans_input',
                        action='store_true', default=False,
                        help='Whether logtrans_input'
                        )
    parser.add_argument('--normalize_amount',
                        action='store_true', default=False,
                        help='Whether normalize_amount'
                        )
    parser.add_argument('--normalize_expression_profile',
                        action='store_true', default=False,
                        help='Whether normalize_expression_profile'
                        )

    parser.add_argument('--graph_AE_epoch', type=int,
                        default=500,
                        help='(int, default 500) Total EM epochs')
    parser.add_argument('--graph_AE_patience', type=int,
                        default=10,
                        help='(int, default 10) Specify how many epochs stop training after validation metrics no longer improve.'
                             ' If the indicator does not improve within the specified epoch number, the learning rate will be adjusted.')
    parser.add_argument('--graph_AE_learning_rate', type=float, default=1e-3,
                        help='(float, default 1e-3) Learning rate')
    parser.add_argument('--graph_AE_factor', type=float, default=1e-1,
                        help='(float, default 1e-1) The learning rate scaling factor, which is used to resize the learning rate. '
                             'When the validation metric stops improving, the learning rate is multiplied by this factor')
    parser.add_argument("--shuffle_each_epoch",
                        action='store_true', default=False,
                        help="Whether to shuffle cell/gene indexes at each epoch")
    parser.add_argument('--use_checkpoint',
                        action='store_true', default=False,
                        help='use_checkpoint')
    parser.add_argument('--graph_AE_use_GAT',
                        action='store_true', default=False,
                        help='(boolean, default False) If true, will use GAT for Graph Gene layers; otherwise will use GCN layers')

    # Graph Gene related
    parser.add_argument('--graph_Gene_dropout', type=float, default=0,
                        help='(int, default 0) The dropout probability for GCN or GAT')
    parser.add_argument('--graph_Gene_gat_multi_heads', type=int, default=2,
                        help='(int, default 2)')
    parser.add_argument('--graph_Gene_gat_hid_embed1', type=int,
                        default=62,
                        help='(int, default 62) The dim for hid_embed')
    parser.add_argument('--graph_Gene_gat_hid_embed2', type=int,
                        default=16,
                        help='(int, default 16) Graph Gene embedding size')
    parser.add_argument('--graph_Gene_gcn_hid_embed1', type=int,
                        default=64,
                        help='(int, default 64) Number of units in hidden layer 1.')
    parser.add_argument('--graph_Gene_gcn_hid_embed2', type=int,
                        default=16,
                        help='(int, default 16) Number of units in hidden layer 2.')
    parser.add_argument('--graph_Gene_hid_embed1', type=int,
                        # default=62,
                        default=64,
                        # default=250,
                        # default=15,
                        help='(int, default 62) The dim for hid_embed')
    parser.add_argument('--graph_Gene_hid_embed2', type=int,
                        default=16,
                        # default=10,
                        help='(int, default 16) Graph Gene embedding size')


    # Feature AE related
    parser.add_argument('--feature_AE_dropout', type=float, default=0,
                        help='(int, default 0) The dropout probability.')
    parser.add_argument('--ae_multi_heads_gene', type=int, default=8,
                        help='(int, default 2)')
    parser.add_argument('--ae_multi_heads_cell', type=int, default=8,
                        help='(int, default 2)')
    parser.add_argument('--feature_AE_hidden1', type=int, default=512, help='Number of units in hidden layer 1.')
    parser.add_argument('--feature_AE_hidden2', type=int, default=128, help='Number of units in hidden layer 2.')


    # Graph Cell related
    parser.add_argument('--graph_Cell_dropout', type=float, default=0,
                        help='(int, default 0) The dropout probability for GCN or GAT')
    parser.add_argument('--graph_Cell_gcn_hid_embed1', type=int,
                        default=64,
                        help='(int, default 64) Number of units in hidden layer 1.')
    parser.add_argument('--graph_Cell_gcn_hid_embed2', type=int,
                        default=16,
                        help='(int, default 16) Number of units in hidden layer 2.')
    parser.add_argument('--graph_Cell_gat_hid_embed1', type=int,
                        default=64,
                        help='(int, default 64) The dim for hid_embed')
    parser.add_argument('--graph_Cell_gat_hid_embed2', type=int,
                        default=16,
                        help='(int, default 16) Graph Gene embedding size')
    parser.add_argument('--graph_Cell_gat_multi_heads', type=int, default=2,
                        help='(int, default 2)')
    parser.add_argument('--graph_Cell_hid_embed1', type=int,
                        default=64,
                        help='(int, default 32) Number of units in hidden layer 1.')
    parser.add_argument('--graph_Cell_hid_embed2', type=int,
                        default=16,
                        help='(int, default 16) Number of units in hidden layer 2.')

    # Output related
    parser.add_argument('--output_dir', type=str,
                        default='outputs/',
                        help="(str, default 'outputs/') Folder for storing all the outputs")

    args = parser.parse_args()

    return args
