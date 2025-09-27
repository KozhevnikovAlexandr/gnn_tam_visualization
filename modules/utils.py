import pandas as pd
import numpy as np
import re
import os

def get_baseline_metrics(model_path):
    model_name = os.path.basename(model_path).split('.')[0]
    baseline_df = pd.read_csv(f'./precomputed_data/{model_name}_baseline_metrics.csv')
    return baseline_df.to_dict('records')

def get_node_importance_matrix(model_path):
    model_name = os.path.basename(model_path).split('.')[0]
    node_importance_matrix = np.load(f'./precomputed_data/{model_name}_node_importance.npy')
    return node_importance_matrix

def get_disabled_sensor_metrics(model_path):
    model_name = os.path.basename(model_path).split('.')[0]
    disabled_sensor_metrics_df = pd.read_csv(f'./precomputed_data/{model_name}_disabled_metrics.csv')
    return disabled_sensor_metrics_df


def get_metric_without_node(df_metrics, node_id):
    node_id = node_id-1
    return df_metrics[df_metrics['disabled_sensor'] == node_id]['TPR'].apply(lambda x: round(x, 3)).tolist() 


def get_node_importance(importance_matrix, node_id):
    return importance_matrix[:,node_id-1]

def parse_model_params(model_path):
    model_filename = os.path.basename(model_path)

    m = re.search(r'(?P<n_gnn>\d+)x(?P<n_hidden>\d+)_(?P<gsl_type>[a-z]+)_(?P<dataset>\w+)', model_filename)
    if m is not None:
        return m.groupdict()

    raise ValueError(f'error parsing model filename: {model_path}')
