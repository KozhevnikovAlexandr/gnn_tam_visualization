import os

import pandas as pd
import numpy as np
import argparse

from modules.utils import parse_model_params

Reinartz_TEP_class_names = [
    "Normal Feed", "A/C feed ratio, B composition constant (stream 4)",
    "B composition, A/C ration constant (stream 4)", "D feed temperature (stream 2)", "Reactor cooling water inlet temperature",
    "Condenser cooling water inlet temperature", "A feed loss (stream 1)", "C header pressure loss - reduced availability",
    "A, B, C feed composition (stream 4)", "D feed temperature (stream 2)", "C feed temperature (stream 4)",
    "Reactor cooling water inlet temperature", "Condenser cooling water inlet temperature", "Reaction kinetics",
    "Reactor cooling water valve", "Condencer cooling water valve", "Unknown", "Unknown", "Unknown", "Unknown",
    "Unknown", "A feed (stream 1) temperature", "E feed (stream 3) temperature", "A feed flow (stream 1)",
    "D feed flow (stream 2)", "E feed flow (stream 3)", "A and C feed flow (stream 4)",
    "Reactor cooling water flow", "Condenser cooling water flow"
]

Reith_TEP_class_names = ['Normal Feed'] + [f'Fault #{i}' for i in range(20)]

def parse_args():
    parser = argparse.ArgumentParser(description='compute baseline model metrics')
    parser.add_argument('model_path', type=str, default='./models/gnn1x1024_directed_reinartz_tep.pt')
    return parser.parse_args()


def main():
    args = parse_args()

    params_dict = parse_model_params(args.model_path)

    if params_dict['dataset'] == 'reinartz_tep':
        class_names = Reinartz_TEP_class_names
    elif params_dict['dataset'] == 'rieth_tep':
        class_names = Reith_TEP_class_names
    else:
        raise NotImplementedError()

    node_importance_matrix_path = ('gnn' + params_dict['n_gnn'] + 'x' + params_dict['n_hidden'] + '_' +
                                   params_dict['gsl_type'] + '_' + params_dict['dataset'] + '_node_importance.npy')
    node_importance_matrix_path = os.path.join('./precomputed_data', node_importance_matrix_path)

    disabled_sensor_metrics_path = ('gnn' + params_dict['n_gnn'] + 'x' + params_dict['n_hidden'] + '_' +
                                    params_dict['gsl_type'] + '_' + params_dict['dataset'] + '_disabled_metrics.csv')
    disabled_sensor_metrics_path = os.path.join('./precomputed_data', disabled_sensor_metrics_path)

    if not os.path.exists(node_importance_matrix_path):
        raise FileNotFoundError(f'node importance matrix not found at {node_importance_matrix_path}, calculate it with compute_node_importance.py')

    if not os.path.exists(disabled_sensor_metrics_path):
        raise FileNotFoundError(f'disabled sensor metrics not found at {disabled_sensor_metrics_path}, calculate it with compute_disabled_sensor_metrics.py')

    node_importance = np.load(node_importance_matrix_path)
    metrics = pd.read_csv(disabled_sensor_metrics_path)

    data = {"Description": class_names,
            "Baseline TPR": metrics.loc[metrics.disabled_sensor == -1].iloc[:, 1:].values[0],
            "Fault №": np.arange(len(class_names)),
            "New TPR": metrics.loc[metrics.disabled_sensor == -1].iloc[:, 1:].values[0],
            "Diff": np.zeros((len(class_names), )),
            "Most important nodes": node_importance.argsort(1)[:, -3:].tolist()}


    baseline_metrics_path = ('gnn' + params_dict['n_gnn'] + 'x' + params_dict['n_hidden'] +
                             '_' + params_dict['gsl_type'] + '_' + params_dict['dataset'] + '_baseline_metrics.csv')
    baseline_metrics_path = os.path.join('./precomputed_data', baseline_metrics_path)
    pd.DataFrame(data).to_csv(baseline_metrics_path, index=False)


if __name__ == '__main__':
    main()