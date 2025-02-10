import pandas as pd
import numpy as np


df_metrics = pd.read_csv('disabled_sensor_metrics (4).csv')

def get_metric_without_node(node_id):
    node_id = node_id-1
    return df_metrics[df_metrics['disabled_sensor'] == node_id]['TPR'].apply(lambda x: round(x, 3)).tolist() 


def get_importance(node_id):
    importance_matrix = np.load('class_wise_node_importance_trained.npy')
    return importance_matrix[:,node_id-1]
