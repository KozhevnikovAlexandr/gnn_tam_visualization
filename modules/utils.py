import pandas as pd
import numpy as np
import re
import os

df_metrics = pd.read_csv('./precomputed_data/disabled_sensor_metrics.csv')

def get_baseline_metrics():
    baseline_df = pd.read_csv('./precomputed_data/metrics.csv')
    return baseline_df.to_dict('records')


def get_metric_without_node(node_id):
    node_id = node_id-1
    return df_metrics[df_metrics['disabled_sensor'] == node_id]['TPR'].apply(lambda x: round(x, 3)).tolist() 


def get_node_importance(node_id):
    importance_matrix = np.load('./precomputed_data/class_wise_node_importance.npy')
    return importance_matrix[:,node_id-1]

def parse_model_params(model_path):
    model_filename = os.path.basename(model_path)

    m = re.search(r'(?P<n_gnn>\d+)x(?P<n_hidden>\d+)_(?P<gsl_type>[a-z]+)_(?P<dataset>\w+)', model_filename)
    if m is not None:
        return m.groupdict()

    raise ValueError(f'error parsing model filename: {model_path}')
