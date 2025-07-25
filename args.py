import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_parser():
    parser = argparse.ArgumentParser()

    # -- Data params ---
    parser.add_argument("--dataset", type=str.upper, default="BZ", help="BZ, HZ, TLM")
    parser.add_argument("--lookback", type=int, default=5,help="")
    parser.add_argument("--normalize", type=str2bool, default=True)
    parser.add_argument("--spec_res", type=str2bool, default=False)

    # -- Model params ---
    # 1D conv layer
    parser.add_argument("--kernel_size", type=int, default=7)
    # GAT layers
    parser.add_argument("--use_gatv2", type=str2bool, default=True)
    parser.add_argument("--feat_gat_embed_dim", type=int, default=None)
    parser.add_argument("--time_gat_embed_dim", type=int, default=None)
    # GRU layer
    parser.add_argument("--gru_n_layers", type=int, default=1)
    parser.add_argument("--gru_hid_dim", type=int, default=256)
    # Forecasting Model
    parser.add_argument("--fc_n_layers", type=int, default=2)
    parser.add_argument("--fc_hid_dim", type=int, default=128)
    # Reconstruction Model
    parser.add_argument("--recon_n_layers", type=int, default=2)
    parser.add_argument("--recon_hid_dim", type=int, default=128)
    parser.add_argument("--use_vae", type=str2bool, default=False)
    # Other
    parser.add_argument("--alpha", type=float, default=0.2)

    # --- Train params ---
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--val_split", type=float, default=0.8)
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--init_lr", type=float, default=1e-3)
    parser.add_argument("--shuffle_dataset", type=str2bool, default=False)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--use_cuda", type=str2bool, default=True)
    parser.add_argument("--print_every", type=int, default=20)
    parser.add_argument("--log_tensorboard", type=str2bool, default=False)

    # --- Predictor params ---
    parser.add_argument("--scale_scores", type=str2bool, default=False)
    parser.add_argument("--use_mov_av", type=str2bool, default=False)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--level", type=float, default=0.98)
    parser.add_argument("--q", type=float, default=0.01) #tlm 0.1,0.85
    parser.add_argument("--dynamic_pot", type=str2bool, default=True)

    return parser
