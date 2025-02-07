import streamlit as st
import networkx as nx
from pyvis.network import Network
import torch
import sys
import pandas as pd
import plotly.express as px

from falut_metrics import get_metric_without_node
from utils import get_baseline_metrics, get_node_color_and_shape
from node_info import get_node_info

sys.path.append('/home/akozhevnikov/graphs/gnn-tam')


def create_visualization(nodes_to_zeros=[]):
    model_path = '/home/akozhevnikov/graphs/gnn-tam/saved_models/gnn1.pt'
    model = torch.load(model_path, weights_only=False).cpu()
    adj = torch.relu(model.gsl[0].gsl_layer.A)
    mask = torch.zeros(52, 52)
    mask.fill_(0.0)
    v, id = (adj + torch.rand_like(adj)*0.01).topk(1, 1)
    mask.scatter_(1, id, v.fill_(1))
    adj = adj*mask
    adj = adj * (1 - torch.eye(52))
    
    for node_id in nodes_to_zeros:
        node_id -= 1
        adj[node_id] = torch.zeros_like(adj[node_id])
        adj[:, node_id] = torch.zeros_like(adj[node_id])
    
    net = Network(notebook=True, directed=True, cdn_resources='in_line')
    for node in range(adj.shape[0]):
        color, shape = get_node_color_and_shape(node)
        size = 20 if shape == 'box' else 13
        net.add_node(node, label=str(node+1), physics=False, title=get_node_info(node), color=color, shape=shape, size=size)
    
    for node in range(adj.shape[0]):
        for node2 in range(adj.shape[0]):
            if adj[node, node2]:
                net.add_edge(node, node2, physics=False)
    
    return net

def calculate_metrics(baseline_metrics, turn_off_node=None):
    if turn_off_node is None:
        for i in baseline_metrics:
            i['TPR New'] = i['TPR Baseline']
            i['Diff'] = 0.0
    else:
        metric_without_node = get_metric_without_node(turn_off_node)
        for idx, i in enumerate(baseline_metrics):
            i['TPR New'] = metric_without_node[idx]
            i['Diff'] = round(i['TPR New'] - i['TPR Baseline'], 3)
    return baseline_metrics

def main():
    #st.set_page_config(layout="wide")

    st.title("Graph Visualization with Metrics")
    
    if 'disabled_nodes' not in st.session_state:
        st.session_state.disabled_nodes = []
    
    if 'metrics' not in st.session_state:
        st.session_state.metrics = calculate_metrics(get_baseline_metrics())
    
    all_nodes = list(range(1, 53))
    
    selected_node = st.selectbox(
        "Select a node to disable",
        all_nodes,
        index=None,
        placeholder="Choose a node to disable"
    )
    
    if st.button("Disable Node"):
        if selected_node is not None:
            st.session_state.disabled_nodes = [selected_node]
            st.session_state.metrics = calculate_metrics(get_baseline_metrics(), turn_off_node=selected_node)
        else:
            st.session_state.disabled_nodes = []
            st.session_state.metrics = calculate_metrics(get_baseline_metrics(), turn_off_node=None)
    
    if st.button("Enable All Nodes"):
        st.session_state.disabled_nodes = []
        st.session_state.metrics = calculate_metrics(get_baseline_metrics(), turn_off_node=None)
    
    net = create_visualization(st.session_state.disabled_nodes)
    
    net.save_graph('graph.html')
    with open('graph.html', 'r', encoding='utf-8') as f:
        html = f.read()
    html = f"""
    <div style="width: 100%; height: 100vh;">
        {html}
    </div>
    """
    st.components.v1.html(html, height=800)
    
    st.header("Metrics")
    metrics = st.session_state.metrics
    
    cols = st.columns(5)
    headers = ["Fault №", "Description", "TPR Baseline", "TPR New", "Diff"]
    for col, header in zip(cols, headers):
        col.write(f"**{header}**")
    
    for metric in metrics:
        cols = st.columns(5)
        for i, header in enumerate(headers):
            cols[i].write(metric[header])
    
    st.header("TPR Distribution Comparison")
    df = pd.DataFrame(st.session_state.metrics)
    df['diff_minus'] = -df['Diff']
    
    fig = px.bar(
        df,
        x='Fault №',
        y=['diff_minus'],
        barmode='overlay',
        title='Comparison of TPR Baseline vs TPR New',
        labels={'value': 'TPR Value', 'variable': 'Metric Type'},
        hover_data=['Description']
    )
    
    fig.update_layout(
        xaxis_title='Fault Number',
        yaxis_title='TPR',
        xaxis={'type': 'category'},
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
if __name__ == "__main__":
    main()