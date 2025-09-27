# 🧠 An Interactive Framework for Interpretable Fault Diagnosis with Graph Neural Networks

> **Code accompanying the paper**: *"An Interactive Framework for Interpretable Fault Diagnosis with Graph Neural Networks"*  

This repository provides an **interactive visualization and analysis framework** for interpretable Fault Detection and Diagnosis (FDD) using Graph Neural Networks (GNNs) with **Trainable Adjacency Matrices (GNN-TAM)**. The system enables users to:

- 🖥️ **Visualize the learned sensor dependency graph**  
- 🎯 **Identify critical sensors for specific faults** via precomputed node importance  
- ⚙️ **Simulate sensor failures** and observe their impact on diagnostic performance  
- 📊 **Compare performance metrics** (TPR) before and after sensor disabling  

Validated on **Tennessee Eastman Process (TEP)** benchmarks, this tool bridges the gap between high-accuracy deep learning models and operational trust in industrial settings.

---

## 🕹️ Installation

```bash
git clone https://github.com/KozhevnikovAlexandr/gnn-tam-visualization.git
cd gnn-tam-visualization
git submodule update --init --recursive
conda env create -f environment.yml
```

---


## 🚀 Run with provided models/data

```bash
conda activate gnn_tam_vis
streamlit run app.py
```

---

## 🔮 Run with custom models/data

Train your model and put it into ```./models```:
```bash
python ./modules/gnn-tam/train.py
cp ./modules/gnn-tam/saved_models/YOUR_MODEL.pt ./models
```

Precompute necessary data:
```bash
python ./compute_node_importance.py YOUR_MODEL_PATH
python ./compute_disabled_sensor_metrics.py YOUR_MODEL_PATH
python ./compute_baseline_metrics.py YOUR_MODEL_PATH
```

Run the app

```bash
conda activate gnn_tam_vis
streamlit run app.py
```

---

## 🎨 Interactive Features

### 🔍 Graph Visualization
- View the **learned adjacency matrix** as a directed graph.
- Nodes are **color-coded** by physical meaning.
- Nodes are **shaped** by type:
  - **Box (□)** = Measured Variable (e.g., sensor readings)
  - **Circle (○)** = Manipulated Variable (e.g., valve positions)
- Display only the **top-K strongest outgoing edges** per node (configurable via slider/dropdown).

### 🛑 Simulate Sensor Failure
- Select any sensor (node) and click **“Disable Node”**.
- The system will:
  - ⚡ **Zero out** all incoming and outgoing edges for that node.
  - 📊 **Recalculate diagnostic performance (TPR)** for all fault types.
  - 📉 Show the **difference (ΔTPR = New TPR – Baseline TPR)** for each fault.
  - 🔴 Visually **highlight the disabled node in red (▲ triangle)** on the graph.

### 📈 Performance Analysis
- **Bar Chart**: Side-by-side comparison of:
  - `Baseline TPR` (🔵 blue) — performance with all sensors active.
  - `New TPR` (🔴 red) — performance after disabling the selected sensor.
- **Interactive Data Table** includes:
  - `Fault №` & `Description`
  - `Most important nodes` for diagnosing this fault
  - `Baseline TPR`, `New TPR`, `ΔTPR`
  - `Disabled Node Importance` — precomputed importance score of the disabled node for *this specific fault*

### ℹ️ Node Information (on Hover)
Hover over any node to see a detailed tooltip with:
- **Sensor name** (e.g., “Reactor Temperature”, “Stripper Level”)
- **Top-3 most significantly affected fault types**, ranked by importance.

---

## 📷 Video

https://www.youtube.com/watch?v=1958k_KO1_4
