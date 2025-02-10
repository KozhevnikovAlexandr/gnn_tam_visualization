import streamlit as st
from pyvis.network import Network
import torch
import sys
import pandas as pd
import plotly.express as px

from falut_metrics import get_importance, get_metric_without_node
from utils import get_baseline_metrics, get_node_color_and_shape, colors, shapes
from node_info import get_node_info

sys.path.append('/home/akozhevnikov/graphs/gnn-tam')

top_k_nodes_base = 2

def create_visualization(nodes_to_zeros=[], top_k_nodes=2):
    device = torch.device("cpu")

    model_path = '/home/akozhevnikov/graphs/gnn_tam_visualization/gnn1_directed.pt'
    model = torch.load(model_path, map_location=device, weights_only=False).eval()
    n_nodes = model.idx.size(0)
    idx = torch.arange(n_nodes, device=device)
    gsl_module = model.gsl[0]

    with torch.no_grad():
        adj = gsl_module.gsl_layer(idx).to(device)
        mask = torch.zeros_like(adj).to(device)
        
        v, indices = (adj).topk(top_k_nodes, dim=1)
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
        node_info = get_node_info(node)
        if node+1 in nodes_to_zeros:
            node_info = f'This node is disabled\n\n{node_info}'
            color = 'red'
            shape = 'triangle'
        
        net.add_node(node, label=str(node+1), physics=True, title=node_info, color=color, shape=shape, size=size)
    
    for node in range(adj.shape[0]):
        for node2 in range(adj.shape[0]):
            if adj[node, node2]:
                net.add_edge(node, node2, physics=True)
    net.show_buttons(filter_=['physics'])
    
    return net

def calculate_metrics(baseline_metrics, turn_off_node=None):
    if turn_off_node is None:
        for i in baseline_metrics:
            i['TPR New'] = i['TPR Baseline']
            i['Diff'] = 0.0
            i['Disabled Node Importance'] = 0.0
    else:
        metric_without_node = get_metric_without_node(turn_off_node)
        imp = get_importance(turn_off_node)
        for idx, i in enumerate(baseline_metrics):
            i['TPR New'] = metric_without_node[idx]
            i['Diff'] = round(i['TPR New'] - i['TPR Baseline'], 3)
            i['Disabled Node Importance'] = float(imp[idx])
    return baseline_metrics

def main():
    st.set_page_config(layout="wide")

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

    selected_k = st.selectbox(
        "Select top k important edges to show",
        range(1, 6),
        index=None,
        placeholder="Choose k param"
    )

    if st.button("Change k"):
        top_k_nodes=selected_k
    else:
        top_k_nodes=top_k_nodes_base
    
    flag = False
    if st.button("Disable Node"):
        flag = True
        if selected_node is not None:
            st.session_state.disabled_nodes = [selected_node]
            st.session_state.metrics = calculate_metrics(get_baseline_metrics(), turn_off_node=selected_node)
        else:
            st.session_state.disabled_nodes = []
            st.session_state.metrics = calculate_metrics(get_baseline_metrics(), turn_off_node=None)
    
    if st.button("Enable All Nodes"):
        st.session_state.disabled_nodes = []
        st.session_state.metrics = calculate_metrics(get_baseline_metrics(), turn_off_node=None)
    
    net = create_visualization(st.session_state.disabled_nodes, top_k_nodes=top_k_nodes)
    
    net.save_graph('graph.html')
    with open('graph.html', 'r', encoding='utf-8') as f:
        html = f.read()
    html = f"""
    <div style="width: 100%; height: 100vh;">
        {html}
    </div>
    """
    st.components.v1.html(html, height=800)

    with st.expander("LEGEND", expanded=True):
        col1, col2, col3 = st.columns([2,2,1])
        
        with col1:
            st.markdown("**Node Colors**")
            for color_key, color_value in colors.items():
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin: -10px 0;">
                    <div style="width: 20px; height: 20px; background-color: {color_value}; 
                             margin-right: 10px; border: 1px solid #666;"></div>
                    <span>{['Flow rate','Pressure','Level','Temperature','Composition','Other'][color_key]}</span>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("**Node Shapes**")
            for shape_key, shape_value in shapes.items():
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
    
    if flag:
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
            hovermode='x unified',
            showlegend=False,
        )
    
    else:
        st.header("TPR Distribution Comparison")
        baseline = pd.DataFrame(get_baseline_metrics())
        fig = px.bar(
            baseline,
            x='Fault №',
            y=['TPR Baseline'],
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
    st.plotly_chart(fig, use_container_width=True)

    columns_to_show = ['Fault №', 'Description', 'Most important nodes', 'TPR Baseline', 'TPR New', 'Diff']
    df_to_show = pd.DataFrame(st.session_state.metrics)[columns_to_show]
    st.dataframe(df_to_show, use_container_width=True)
    
if __name__ == "__main__":
    main()