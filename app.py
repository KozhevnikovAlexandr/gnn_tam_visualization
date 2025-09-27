import streamlit as st
from pyvis.network import Network
import torch
import pandas as pd
import plotly.express as px
import importlib
import os
import sys

from modules.utils import (get_baseline_metrics, get_node_importance, get_metric_without_node, 
                           parse_model_params, get_disabled_sensor_metrics, get_node_importance_matrix)
from modules.node_info import get_node_info, get_node_color_and_shape, COLORS, SHAPES
from modules.dataset_info import DATASET_INFO

# questionable but okay ig since gnn-tam has a dash in name and we cannot alter it
original_path = sys.path.copy()
gnn_tam_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'modules', 'gnn-tam'))

if gnn_tam_dir not in sys.path:
    sys.path.insert(0, gnn_tam_dir)

GNN_TAM = getattr(importlib.import_module('modules.gnn-tam.gnn'), 'GNN_TAM')
if GNN_TAM is None:
    raise ImportError('GNN_TAM not found in modules.gnn-tam.gnn')

sys.path[:] = original_path

if 'top_k_nodes' not in st.session_state:
    st.session_state.top_k_nodes = 2
st.set_page_config(layout="wide")

st.title("Graph Visualization with Metrics")

def create_visualization(params_dict, nodes_to_zeros=[], top_k_nodes=2):
    device = torch.device("cpu")

    model = GNN_TAM(n_nodes=DATASET_INFO[params_dict['dataset']]['n_sensors'],
                    window_size=100,
                    n_classes=DATASET_INFO[params_dict['dataset']]['n_classes'],
                    n_gnn=int(params_dict['n_gnn']),
                    gsl_type=params_dict['gsl_type'],
                    n_hidden=int(params_dict['n_hidden']),
                    device=device)
    model.load_state_dict(torch.load(st.session_state.model_path, map_location=device, weights_only=False))
    model.eval()
    n_nodes = model.idx.size(0)
    idx = torch.arange(n_nodes, device=device)
    gsl_module = model.gsl[0]

    with torch.no_grad():
        adj = gsl_module.gsl_layer(idx).to(device)
        mask = torch.zeros_like(adj).to(device)
        
        v, indices = adj.topk(top_k_nodes, dim=1)
        indices = indices.to(device)
        mask.scatter_(1, indices, 1)
        adj = adj * mask
        adj = adj * (1 - torch.eye(n_nodes, device=device))
        
    for node_id in nodes_to_zeros:
        node_id -= 1
        adj[node_id] = torch.zeros_like(adj[node_id])
        adj[:, node_id] = torch.zeros_like(adj[node_id])
    
    net = Network(notebook=True, directed=True, cdn_resources='in_line', neighborhood_highlight=True, select_menu=False)
    for node in range(adj.shape[0]):
        color, shape = get_node_color_and_shape(node)
        size = 20 if shape == 'box' else 13
        node_info = get_node_info(st.session_state.baseline_metrics, st.session_state.node_importance_matrix, node)
        if node+1 in nodes_to_zeros:
            node_info = f'This node is disabled\n\n{node_info}'
            color = 'red'
            shape = 'triangle'
        
        net.add_node(node, label=str(node+1), physics=False, title=node_info, color=color, shape=shape, size=size, font={"size": 12})
    
    for node in range(adj.shape[0]):
        for node2 in range(adj.shape[0]):
            if adj[node, node2]:
                net.add_edge(node, node2, physics=False)
    
    return net

def calculate_metrics(baseline_metrics, turn_off_node=None):
    if turn_off_node is None:
        for i in baseline_metrics:
            i['New TPR'] = i['Baseline TPR']
            i['Diff'] = 0.0
            i['Disabled Node Importance'] = 0.0
    else:
        metric_without_node = get_metric_without_node(st.session_state.disabled_sensor_metrics, turn_off_node)
        imp = get_node_importance(st.session_state.node_importance_matrix, turn_off_node)
        for idx, i in enumerate(baseline_metrics):
            i['New TPR'] = metric_without_node[idx]
            i['Diff'] = round(i['New TPR'] - i['Baseline TPR'], 3)
            i['Disabled Node Importance'] = float(imp[idx])
    return baseline_metrics

def main():
    if 'model_path' not in st.session_state:
        st.session_state.model_path = None

    model_options = ['Select model'] + [e for e in os.listdir('./models') if e.endswith('.pt')]
    model_name = st.selectbox('Please select a model to run a simulation on', model_options)
    if model_name != 'Select model':
        st.session_state.model_path = os.path.join('./models/', model_name)
    else:
        st.session_state.selected_value = None

    if st.session_state.model_path is not None:
        st.session_state.model_params_dict = parse_model_params(st.session_state.model_path)
        st.session_state.disabled_nodes = []
        st.session_state.baseline_metrics = calculate_metrics(get_baseline_metrics(st.session_state.model_path))
        st.session_state.metrics = st.session_state.baseline_metrics
        st.session_state.node_importance_matrix = get_node_importance_matrix(st.session_state.model_path)
        st.session_state.disabled_sensor_metrics = get_disabled_sensor_metrics(st.session_state.model_path)

        all_nodes = list(range(1, 53))

        selected_node = st.selectbox(
            "Select a node to disable",
            all_nodes,
            index=None,
            placeholder="Choose a node to disable"
        )

        selected_k = st.selectbox(
            "Select top k important edges to show",
            range(1, 6),
            index=None,
            placeholder="Choose k param",
        )

        col1, col2, col3 = st.columns(3)
        node_disabled_flag = False
        with col1:
            if st.button("Change k") and selected_k is not None:
                st.session_state.top_k_nodes = selected_k

        with col2:
            if st.button("Disable Node"):
                node_disabled_flag = True
                if selected_node is not None:
                    st.session_state.disabled_nodes = [selected_node]
                    st.session_state.metrics = calculate_metrics(st.session_state.baseline_metrics, turn_off_node=selected_node)
                else:
                    st.session_state.disabled_nodes = []
                    st.session_state.metrics = calculate_metrics(st.session_state.baseline_metrics, turn_off_node=None)

        with col3:
            if st.button("Enable All Nodes"):
                st.session_state.disabled_nodes = []
                st.session_state.metrics = calculate_metrics(st.session_state.baseline_metrics, turn_off_node=None)

            net = create_visualization(st.session_state.model_params_dict, st.session_state.disabled_nodes,
                                       top_k_nodes=st.session_state.top_k_nodes)
            options = """
            var options = {
            "edges": {
                "smooth": false
            }
            }
            """
            net.set_options(options)

        html = f"""
        <div style="width: 100%; height: 100%">
            {net.generate_html()}
        </div>
        """
        st.components.v1.html(html, height=610)

        with st.expander("LEGEND", expanded=True):
            col1, col2, col3 = st.columns([2,2,1])

            with col1:
                st.markdown("**Node Colors**")
                for color_key, color_value in COLORS.items():
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; margin: -10px 0;">
                        <div style="width: 20px; height: 20px; background-color: {color_value}; 
                                 margin-right: 10px; border: 1px solid #666;"></div>
                        <span>{['Flow rate','Pressure','Level','Temperature','Composition','Other'][color_key]}</span>
                    </div>
                    """, unsafe_allow_html=True)

            with col2:
                st.markdown("**Node Shapes**")
                for shape_key, shape_value in SHAPES.items():
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; margin: -10px 0;">
                        <div style="width: 20px; height: 20px; margin-right: 10px;
                                 display: flex; justify-content: center; align-items: center;">
                            <div style="width: 15px; height: 15px; 
                                     border: 2px solid #333; background: {'none' if shape_value == 'box' else '#333'};
                                     border-radius: {'0%' if shape_value == 'box' else '50%'};"></div>
                        </div>
                        <span>{['Process measurement','Manipulated Variables'][shape_key]}</span>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("""
                <div style="display: flex; align-items: center; margin: -10px 0;">
                    <div style="width: 20px; height: 20px; margin-right: 10px;
                             display: flex; justify-content: center; align-items: center;">
                        <span style="color: red; font-size: 24px;">▲</span>
                    </div>
                    <span style="color: red;">Disabled node</span>
                </div>
                """, unsafe_allow_html=True)

        if node_disabled_flag:
            st.header("TPR Distribution Comparison")
            df = pd.DataFrame(st.session_state.metrics)
            df['diff_minus'] = -df['Diff']

            fig = px.bar(
                df,
                x='Fault №',
                y=['Baseline TPR', 'New TPR'],
                barmode='overlay',
                title='Comparison of Baseline TPR vs New TPR',
                labels={'value': 'TPR Value', 'variable': 'Metric Type'},
                hover_data=['Description'],
                color_discrete_map={
                    'Baseline TPR': 'blue',
                    'New TPR': 'red'
                }
            )

            fig.update_layout(
                xaxis_title='Fault Number',
                yaxis_title='TPR',
                xaxis={'type': 'category'},
                hovermode='x unified',
                showlegend=True,
            )

        else:
            st.header("TPR Distribution Comparison")
            baseline = pd.DataFrame(st.session_state.baseline_metrics)
            fig = px.bar(
                baseline,
                x='Fault №',
                y=['Baseline TPR'],
                barmode='overlay',
                title='TPR for each fault type with all nodes',
                labels={'value': 'TPR Value', 'variable': 'Metric Type'},
                hover_data=['Description']
            )

            fig.update_layout(
                xaxis_title='Fault Number',
                yaxis_title='TPR',
                xaxis={'type': 'category'},
                hovermode='x unified',
                showlegend=False,
            )
        st.plotly_chart(fig, width='stretch')

        columns_to_show = ['Fault №', 'Description', 'Most important nodes', 'Baseline TPR', 'New TPR', 'Diff']
        df_to_show = pd.DataFrame(st.session_state.metrics)[columns_to_show]
        st.dataframe(df_to_show, width='stretch')
    else:
        st.header("Please select a model")

if __name__ == "__main__":
    main()