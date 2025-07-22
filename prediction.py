import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader



class Predictor:
    """DD-GAT predictor .

    :param model: DD-GAT model (pre-trained) used to forecast and reconstruct
    :param window_size: Length of the input sequence
    :param n_features: Number of input features
    :param pred_args: params for thresholding and predicting anomalies

    """

    def __init__(self, model, window_size, n_features, pred_args, summary_file_name="summary.txt"):
        self.model = model
        self.window_size = window_size
        self.n_features = n_features
        self.dataset = pred_args["dataset"]
        self.target_dims = pred_args["target_dims"]
        self.scale_scores = pred_args["scale_scores"]
        self.q = pred_args["q"]
        self.level = pred_args["level"]
        self.dynamic_pot = pred_args["dynamic_pot"]
        self.use_mov_av = pred_args["use_mov_av"]
        self.gamma = pred_args["gamma"]
        self.reg_level = pred_args["reg_level"]
        self.save_path = pred_args["save_path"]
        self.use_vae  = pred_args["use_vae"]
        self.batch_size = 256
        self.use_cuda = True
        self.pred_args = pred_args
        self.summary_file_name = summary_file_name

    def get_score(self, values):
  
        data = SlidingWindowDataset(values, self.window_size, self.target_dims)
        loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size, shuffle=False)
        device = "cuda" if self.use_cuda and torch.cuda.is_available() else "cpu"

        self.model.eval()
        preds = []
        recons = []
        with torch.no_grad():
            for x, y in tqdm(loader):
                x = x.to(device)
                y = y.to(device)

                y_hat, _,_,_ = self.model(x)
                recon_x = torch.cat((x[:, 1:, :], y), dim=1)
                _, window_recon,_,_ = self.model(recon_x)

                preds.append(y_hat.detach().cpu().numpy())
                if self.use_vae:
                    recons.append(window_recon.detach().cpu().numpy())
                else:
                    recons.append(window_recon[:, -1, :].detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        recons = np.concatenate(recons, axis=0)
        actual = values.detach().cpu().numpy()[self.window_size:]

        if self.target_dims is not None:
            actual = actual[:, self.target_dims]

        anomaly_scores = np.zeros_like(actual)
        df_dict = {}
        for i in range(preds.shape[1]):
            df_dict[f"Forecast_{i}"] = preds[:, i]
            df_dict[f"Recon_{i}"] = recons[:, i]
            df_dict[f"True_{i}"] = actual[:, i]
            if self.use_vae:
                a_score = ((preds[:, i] - actual[:, i]) ** 2 + self.gamma * (1-recons[:, i]))/( 1+ self.gamma)
            else:
                a_score = np.sqrt((preds[:, i] - actual[:, i]) ** 2) + self.gamma * np.sqrt(
                    (recons[:, i] - actual[:, i]) ** 2)

            if self.scale_scores:
                q75, q25 = np.percentile(a_score, [75, 25])
                iqr = q75 - q25
                median = np.median(a_score)
                a_score = (a_score - median) / (1+iqr)

            anomaly_scores[:, i] = a_score
            df_dict[f"A_Score_{i}"] = a_score

        df = pd.DataFrame(df_dict)
        anomaly_scores = np.mean(anomaly_scores, 1)
        df['A_Score_Global'] = anomaly_scores

        return df

    def predict_anomalies(self, train, test, true_anomalies, load_scores=False, save_output=True,
                          scale_scores=False):
        """ Predicts anomalies

        """

        if load_scores:
            print("Loading anomaly scores")
            pass

        print("-- Done.")
