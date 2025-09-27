from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import random_split
from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import argparse
import sys
import os
import importlib

from modules.dataset_info import DATASET_INFO
from modules.utils import parse_model_params
from fddbenchmark import FDDDataset, FDDDataloader, FDDEvaluator

original_path = sys.path.copy()
gnn_tam_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'modules', 'gnn-tam'))

if gnn_tam_dir not in sys.path:
    sys.path.insert(0, gnn_tam_dir)

GNN_TAM = getattr(importlib.import_module('modules.gnn-tam.gnn'), 'GNN_TAM')
if GNN_TAM is None:
    raise ImportError('GNN_TAM not found in modules.gnn-tam.gnn')

sys.path[:] = original_path


def parse_args():
    parser = argparse.ArgumentParser(description='precompute model metrics with iteratively disabled sensors')
    parser.add_argument('model_path', type=str, default='./models/gnn1x1024_directed_reinartz_tep.pt')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--n_batches', type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    print('using device:', device)

    params_dict = parse_model_params(args.model_path)

    dataset = FDDDataset(name=params_dict['dataset'])
    scaler = StandardScaler()
    scaler.fit(dataset.df[dataset.train_mask])
    dataset.df[:] = scaler.transform(dataset.df)

    test_dl = FDDDataloader(
            dataframe=dataset.df,
            label=dataset.label,
            mask=dataset.test_mask,
            window_size=100,
            step_size=1,
            use_minibatches=True,
            batch_size=args.batch_size,
            shuffle=True
        )

    model = GNN_TAM(n_nodes=DATASET_INFO[params_dict['dataset']]['n_sensors'],
                    window_size=100,
                    n_classes=DATASET_INFO[params_dict['dataset']]['n_classes'],
                    n_gnn=int(params_dict['n_gnn']),
                    gsl_type=params_dict['gsl_type'],
                    n_hidden=int(params_dict['n_hidden']),
                    device=device)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=False))
    model.to(device)
    model.eval()

    overall_metrics = pd.DataFrame(columns=['disabled_sensor'] + [f'state_{i}' for i in range(DATASET_INFO[params_dict['dataset']]['n_classes'])])

    preds = []
    test_labels = []
    for test_ts, test_index, test_label in tqdm(test_dl, total=args.n_batches):
        ts = torch.FloatTensor(test_ts).to(device)
        ts = torch.transpose(ts, 1, 2)
        logits = model(ts).detach()
        pred = logits.argmax(axis=1).cpu().numpy()
        preds.append(pd.Series(pred, index=test_index))
        test_labels.append(pd.Series(test_label, index=test_index))

        if len(preds) > args.n_batches:
            break
    pred = pd.concat(preds)
    test_label = pd.concat(test_labels)

    evaluator = FDDEvaluator(step_size=1)
    metrics = evaluator.evaluate(test_label, pred)['classification']['TPR']
    overall_metrics.loc[overall_metrics.shape[0]] = np.concatenate([np.array([-1]), metrics])

    for sensor_idx in trange(DATASET_INFO[params_dict['dataset']]['n_sensors']):
        preds = []
        test_labels = []
        for test_ts, test_index, test_label in test_dl:
            ts = torch.FloatTensor(test_ts).to(device)
            ts = torch.transpose(ts, 1, 2)
            logits = model.forward_disabled_sensor(ts, sensor_idx).detach()
            pred = logits.argmax(axis=1).cpu().numpy()
            preds.append(pd.Series(pred, index=test_index))
            test_labels.append(pd.Series(test_label, index=test_index))

            if len(preds) > args.n_batches:
                break
        pred = pd.concat(preds)
        test_label = pd.concat(test_labels)

        evaluator = FDDEvaluator(step_size=1)
        metrics = evaluator.evaluate(test_label, pred)['classification']['TPR']
        overall_metrics.loc[overall_metrics.shape[0]] = np.concatenate([np.array([sensor_idx]), metrics])

    fn = f'{os.path.basename(args.model_path).split(".")[0]}_disabled_metrics.csv'

    state_dfs = list()
    for state in filter(lambda x: 'state' in x, overall_metrics.columns):
        state_df = overall_metrics[['disabled_sensor', state]].copy()
        state_df['state_id'] = int(state.split('_')[1])
        state_df = state_df.rename(columns={state: 'TPR'})
        state_dfs.append(state_df)

    overall_metrics = pd.concat(state_dfs)[['disabled_sensor', 'state_id', 'TPR']].sort_values(['disabled_sensor', 'state_id'])

    overall_metrics.to_csv(os.path.join('./precomputed_data/', fn), index=False)


if __name__ == '__main__':
    main()