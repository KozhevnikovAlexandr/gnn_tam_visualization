import numpy as np

from utils import get_baseline_metrics

importance_matrix = np.load('class_wise_node_importance_trained.npy')
baseline_metrics = get_baseline_metrics()

node_names = ['A feed (stream 1)',
 'D feed (stream 2)',
 'E feed (stream 3)',
 'A and C feed (stream 4)',
 'Recycle flow (stream 8)',
 'Reactor feed rate (stream 6)',
 'Reactor pressure',
 'Reactor level',
 'Reactor temperature',
 'Purge rate (stream 9)',
 'Product separator temperature',
 'Product separator level',
 'Product separator pressure',
 'Product separator underflow (stream 10)',
 'Stripper level',
 'Stripper pressure',
 'Stripper underflow (stream 11)',
 'Stripper temperature',
 'Stripper steam flow',
 'Compressor work',
 'Reactor cooling water outlet temperature',
 'Stripper temperature',
 'Reactor feed analysis (Component A)',
 'Reactor feed analysis (Component B)',
 'Reactor feed analysis Component C)',
 'Reactor feed analysis (Component D)',
 'Reactor feed analysis (Component E)',
 'Reactor feed analysis (Component F)',
 'Purge gas analysis (Component A)',
 'Purge gas analysis (Component B)',
 'Purge gas analysis (Component C)',
 'Purge gas analysis (Component D)',
 'Purge gas analysis (Component E)',
 'Purge gas analysis (Component F)',
 'Purge gas analysis (Component G)',
 'Purge gas analysis (Component H)',
 'Product analysis (Component D)',
 'Product analysis (Component E)',
 'Product analysis (Component F)',
 'Product analysis (Component G)',
 'Product analysis (Component H)',
 'D feed flow (stream 2)',
 'E feed flow (stream 3)',
 'A feed flow (stream 1)',
 'A and C feed flow (stream 4)',
 'Compressor recycle valve',
 'D feed flow (stream 2)',
 'D feed flow (stream 2)',
 'D feed flow (stream 2)',
 'D feed flow (stream 2)',
 'D feed flow (stream 2)',
 'D feed flow (stream 2)']


def get_node_info(node_id, topk=3):
    result = f'{node_names[node_id]} \n \n Most significant faults:'
    top_importance = np.argpartition(importance_matrix[:,node_id], -topk)[-topk:].tolist()
    for ind, i in enumerate(top_importance):
        result += f'\n {i}. {baseline_metrics[i]["Description"]} -- {round(float(importance_matrix[:,node_id][i]), 2)}'
    return result