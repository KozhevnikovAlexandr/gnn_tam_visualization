from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from tqdm import tqdm, trange
import numpy as np
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
    return parser.parse_args()


class GuidedBackprop():
    def __init__(self, model):
        self.model = model
        self.input_reconstruction = None # store R0
        self.activation_maps = []  # store f1, f2, ... 
        self.model.eval()
        self.register_hooks()

    def register_hooks(self):
        def first_layer_hook_fn(module, grad_in, grad_out):
            self.input_reconstruction = grad_out[0]

        def forward_hook_fn(module, input, output):
            self.activation_maps.append(output)

        def backward_hook_fn(module, grad_in, grad_out):
            grad = self.activation_maps.pop()
            # for the forward pass, after the ReLU operation,
            # if the output value is positive, we set the value to 1,
            # and if the output value is negative, we set it to 0.
            grad[grad > 0] = 1

            # grad_out[0] stores the gradients for each feature map,
            # and we only retain the positive gradients
            positive_grad_out = torch.clamp(grad_out[0], min=0.0)
            new_grad_in = positive_grad_out * grad

            return (new_grad_in,)

        modules = list(self.model.modules())

        # travese the modules，register forward hook & backward hook
        # for the ReLU
        num_of_relus_registered = 0
        
        for module in modules:
            if isinstance(module, nn.ReLU):
                module.register_forward_hook(forward_hook_fn)
                module.register_full_backward_hook(backward_hook_fn)
                num_of_relus_registered += 1

        print('# hooks registered:', num_of_relus_registered)

        # register backward hook for the first conv layer
        first_layer = self.model.gsl[0]
        first_layer.register_full_backward_hook(first_layer_hook_fn)

    def visualize(self, input_data, target_classes, device):
        model_output = self.model(input_data)
        self.model.zero_grad()
        pred_class = model_output.argmax().item()

        grad_target_map = torch.zeros(model_output.shape,
                                      dtype=torch.float,
                                      device=device)
        if target_classes is not None:
            grad_target_map[torch.arange(model_output.shape[0]), target_classes] = 1
        else:
            grad_target_map[torch.arange(model_output.shape[0]), pred_class] = 1

        model_output.backward(grad_target_map)

        result = self.input_reconstruction.data
        return result.detach().cpu().numpy()


def get_adjacency_matrix(model):
    adj = model.gsl[0](model.idx)
    adj = adj * model.z

    return adj


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

    guided_bp = GuidedBackprop(model)

    xs = list()
    ys = list()
    for x, index, y in tqdm(test_dl):
        x = torch.as_tensor(x, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.long)
        x = x.transpose(1, 2)

        xs.append(x)
        ys.append(y)

        if len(xs) > 2000:
            break

    xs = torch.cat(xs, dim=0)
    ys = torch.cat(ys, dim=0)

    node_importance = list()

    for c in tqdm(torch.unique(ys)):
        node_importance_ = list()
        cxs = xs[ys == c]
        cys = ys[ys == c]

        for i in tqdm(range(0, cxs.shape[0], args.batch_size), leave=False):
            node_importance_mat = np.abs(guided_bp.visualize(cxs[i: i+args.batch_size].to(device),
                                                             cys[i: i+args.batch_size].to(device), device))
            node_importance_.append((node_importance_mat.mean(axis=0) + node_importance_mat.mean(axis=1)) / 2)
        node_importance_ = np.stack(node_importance_, axis=0).mean(axis=0)
        node_importance.append(node_importance_)

    node_importance = np.abs(np.stack(node_importance))

    top4_sensors_for_each_class = np.argsort(node_importance, axis=-1)[:, -4:]
    heatmap = np.zeros(node_importance.shape, dtype=np.uint8)
    for i in range(4):
        heatmap[np.arange(heatmap.shape[0]), top4_sensors_for_each_class[:, i]] = i

    node_importance_norm = node_importance - node_importance.min(axis=1)[:, None]
    node_importance_norm /= node_importance_norm.max(axis=1)[:, None]

    fn = f'{os.path.basename(args.model_path).split(".")[0]}_node_importance.npy'
    np.save(os.path.join('./precomputed_data/', fn), node_importance_norm)


if __name__ == '__main__':
    main()