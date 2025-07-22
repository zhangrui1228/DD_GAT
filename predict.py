import argparse
import json
import datetime

from args import get_parser, str2bool
from utils import *
from dd_gat import DD_GAT
from prediction import Predictor

if __name__ == "__main__":

    parser = get_parser()
    parser.add_argument("--model_id", type=str, default='-1',
                        help="ID (datetime) of pretrained model to use, '-1' for latest, '-2' for second latest, etc")
    parser.add_argument("--load_scores", type=str2bool, default=False, help="To use already computed anomaly scores")
    parser.add_argument("--save_output", type=str2bool, default=True)
    args = parser.parse_args()
    print(args)

    dataset = args.dataset
    if args.model_id is None:
        if dataset == 'SMD':
            dir_path = f"./output/{dataset}/{args.group}"
        else:
            dir_path = f"./output/{dataset}"
        dir_content = os.listdir(dir_path)
        subfolders = [subf for subf in dir_content if os.path.isdir(f"{dir_path}/{subf}") and subf != "logs"]
        date_times = [datetime.datetime.strptime(subf, '%d%m%Y_%H%M%S') for subf in subfolders]
        date_times.sort()
        model_datetime = date_times[-1]
        model_id = model_datetime.strftime('%d%m%Y_%H%M%S')

    else:
        if args.model_id.startswith('-'):
            dir_content = os.listdir(f"./output/{dataset}")
            datetimes = [datetime.datetime.strptime(subf, '%d%m%Y_%H%M%S') for subf in dir_content if os.path.isdir(f"./output/{dataset}/{subf}")
                          and subf not in ['logs']]
            datetimes.sort()
            model_id = datetimes[int(args.model_id)].strftime('%d%m%Y_%H%M%S')
            print(f"Using model {model_id} from {dataset} dataset")

    if dataset == "SMD":
        model_path = f"./output/{dataset}/{args.group}/{model_id}"
    elif dataset in ['BZ', 'HZ','TLM']:
        model_path = f"./output/{dataset}/{model_id}"
    else:
        raise Exception(f'Dataset "{dataset}" not available.')

    # Check that model exist
    if not os.path.isfile(f"{model_path}/model.pt"):
        raise Exception(f"<{model_path}/model.pt> does not exist.")

    # Get configs of model
    print(f'Using model from {model_path}')
    model_parser = argparse.ArgumentParser()
    model_args, unknown = model_parser.parse_known_args()
    model_args_path = f"{model_path}/config.txt"

    with open(model_args_path, "r") as f:
        model_args.__dict__ = json.load(f)
    window_size = model_args.lookback

    # Check that model is trained on specified dataset
    if args.dataset.lower() != model_args.dataset.lower():
        raise Exception(f"Model trained on {model_args.dataset}, but asked to predict {args.dataset}.")

    elif args.dataset == "SMD" and args.group != model_args.group:
        print(f"Model trained on SMD group {model_args.group}, but asked to predict SMD group {args.group}.")

    window_size = model_args.lookback
    normalize = model_args.normalize
    n_epochs = model_args.epochs
    batch_size = model_args.bs
    init_lr = model_args.init_lr
    val_split = model_args.val_split
    shuffle_dataset = model_args.shuffle_dataset
    use_cuda = model_args.use_cuda
    print_every = model_args.print_every
    group_index = model_args.group[0]
    index = model_args.group[2:]
    args_summary = str(model_args.__dict__)

    (x_train, _), (x_test, y_test) = get_data(args.dataset, normalize=normalize)

    x_train = torch.from_numpy(x_train).float()
    x_test = torch.from_numpy(x_test).float()
    n_features = x_train.shape[1]

    target_dims = get_target_dims(args.dataset)
    print(f"Will forecast and reconstruct input features: {target_dims}")
    if target_dims is None:
        out_dim = n_features
    elif type(target_dims) == int:
        out_dim = 1
    else:
        out_dim = len(target_dims)

    train_dataset = SlidingWindowDataset(x_train, window_size, target_dims)
    test_dataset = SlidingWindowDataset(x_test, window_size, target_dims)

    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, batch_size, val_split, shuffle_dataset, test_dataset=test_dataset
    )

    train_dataset = SlidingWindowDataset(x_train, window_size, target_dims)
    test_dataset = SlidingWindowDataset(x_test, window_size, target_dims)

    model = DD_GAT(
        n_features,
        window_size,
        out_dim,
        kernel_size=model_args.kernel_size,
        use_gatv2=model_args.use_gatv2,
        feat_gat_embed_dim=model_args.feat_gat_embed_dim,
        time_gat_embed_dim=model_args.time_gat_embed_dim,
        gru_n_layers=model_args.gru_n_layers,
        gru_hid_dim=model_args.gru_hid_dim,
        forecast_n_layers=model_args.fc_n_layers,
        forecast_hid_dim=model_args.fc_hid_dim,
        recon_n_layers=model_args.recon_n_layers,
        recon_hid_dim=model_args.recon_hid_dim,
        dropout=model_args.dropout,
        alpha=model_args.alpha,
        use_vae=model_args.use_vae,
    )

    device = "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
    load(model, f"{model_path}/model.pt", device=device)
    model.to(device)

    # Some suggestions for POT args
    level_q_dict = {
        "HZ": (0.95, 0.001),
        "BZ": (0.90, 0.001),
        "TLM": (0.85, 0.1),
    }

    level, q = level_q_dict[args.dataset]
    if args.level is not None:
        level = args.level
    if args.q is not None:
        q = args.q

    # Some suggestions for Epsilon args
    reg_level_dict = { "HZ":1,"BZ":1, "TLM": 1}
    reg_level = reg_level_dict[dataset]
    prediction_args = {
        'dataset': dataset,
        "target_dims": target_dims,
        'scale_scores': args.scale_scores,
        "level": level,
        "q": q,
        'dynamic_pot': args.dynamic_pot,
        "use_mov_av": args.use_mov_av,
        "gamma": args.gamma,
        "reg_level": reg_level,
        "save_path": f"{model_path}",
        "use_vae":args.use_vae,
    }
   
    count = 0
    for filename in os.listdir(model_path):
        if filename.startswith("summary"):
            count += 1
    if count == 0:
        summary_file_name = "summary.txt"
    else:
        summary_file_name = f"summary_{count}.txt"

    label = y_test[window_size:] if y_test is not None else None
    predictor = Predictor(model, window_size, n_features, prediction_args, summary_file_name=summary_file_name)
    predictor.predict_anomalies(x_train, x_test, label,
                                load_scores=args.load_scores,
                                save_output=args.save_output)
