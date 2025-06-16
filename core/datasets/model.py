


import torch
import torch.nn.functional as F
from torch_geometric.nn import JumpingKnowledge

from core.datasets.multihead_attn import GraphormerGraphEncoderLayer


def reshape_features(feature_tensor, node_num):
    """
    Reshape a feature tensor from shape (sum_num_nodes, F) to (B, max_node_num, F).

    Parameters:
    - feature_tensor (torch.Tensor): The input tensor with shape (sum_num_nodes, F).
    - node_num (torch.Tensor): A tensor of shape (B,) indicating the number of nodes in each graph.

    Returns:
    - torch.Tensor: A tensor of shape (B, max_node_num, F), where max_node_num is the maximum number of nodes in any graph.
    """
    B = node_num.shape[0]
    F = feature_tensor.shape[1]
    max_node_num = node_num.max()
    sum_num_nodes = feature_tensor.shape[0]

    graph_ids = torch.repeat_interleave(torch.arange(B, device=feature_tensor.device), node_num)

    start_indices = torch.cat((torch.tensor([0], device=feature_tensor.device), torch.cumsum(node_num, dim=0)[:-1]))
    repeated_start_indices = torch.repeat_interleave(start_indices, node_num)
    node_indices_within_graph = torch.arange(sum_num_nodes, device=feature_tensor.device) - repeated_start_indices

    reshaped_tensor = torch.zeros(B, max_node_num, F, device=feature_tensor.device, dtype=feature_tensor.dtype)

    reshaped_tensor[graph_ids, node_indices_within_graph, :] = feature_tensor

    return reshaped_tensor


def inverse_reshape_features(reshaped_tensor, node_num):
    """
    Convert a feature tensor from shape (B, max_node_num, F) back to (sum_num_nodes, F).

    Parameters:
    - reshaped_tensor (torch.Tensor): The input tensor with shape (B, max_node_num, F).
    - node_num (torch.Tensor): A tensor of shape (B,) indicating the number of nodes in each graph.

    Returns:
    - torch.Tensor: A tensor of shape (sum_num_nodes, F), where sum_num_nodes is the sum of node_num.
    """
    B, max_node_num, F = reshaped_tensor.shape
    sum_num_nodes = node_num.sum().item()

    device = reshaped_tensor.device

    node_indices = torch.arange(max_node_num, device=device).unsqueeze(0).expand(B, max_node_num)
    node_num_expanded = node_num.unsqueeze(1).expand(B, max_node_num)
    mask = node_indices < node_num_expanded

    masked_features = reshaped_tensor[mask]

    return masked_features


class GTranQEM(torch.nn.Module):
    def __init__(self, args, more_args=None):
        super().__init__()
        self.args = args
        self.more_args = more_args
        xlenth = self.args.input_dim
        hidden_dim = self.args.hidden_dim

        print(self.more_args)

        print("hidden_dim", hidden_dim)

        setattr(self, f"conv{0}", GraphormerGraphEncoderLayer(hidden_dim))
        for i in range(1, self.args.num_layers):
            setattr(self, f"conv{i}", GraphormerGraphEncoderLayer(hidden_dim))

        self.linear = torch.nn.Linear(hidden_dim, 1)
        
        self.linear2 = torch.nn.Linear(6, hidden_dim)
        self.linear3 = torch.nn.Linear(xlenth, hidden_dim)
        self.linear4 = torch.nn.Linear(hidden_dim, hidden_dim)

        self.jump = JumpingKnowledge(mode='cat')

        if self.more_args.special_node_with_grad:
            self.special_node_tensor = torch.nn.Parameter(
                torch.normal(0, 1, (1, xlenth)),
                requires_grad=True
            )


    def forward(self, data):
        x, edge_index, noise_y, edge_attr, spatial_pos = data.x, data.edge_index, data.noise_y, data.edge_attr, data.spatial_pos
        node_num = data.num_nodes_tensor
        last_node_idx = torch.cumsum(node_num, dim=0) - 1

        if self.more_args.special_node_with_grad:
            expanded_special_nodes = self.special_node_tensor.expand(len(last_node_idx), -1).to(x.device)
            x = x.clone()
            x[last_node_idx] = expanded_special_nodes
            print("use special node with grad")
            print("special node sum", self.special_node_tensor.sum())

        x = self.linear3(x)
        x = F.relu(x)

        x = reshape_features(x, node_num)
        x = x.permute(1, 0, 2)
        max_node, bs, _ = x.shape

        edge_attr = self.linear2(edge_attr)
        edge_attr = F.relu(edge_attr)
        edge_attr_dim = edge_attr.shape[2]
        edge_attr = edge_attr.reshape(edge_attr.shape[0], -1)
        edge_attr = reshape_features(edge_attr, node_num)
        edge_attr = edge_attr.reshape(bs, max_node, -1, edge_attr_dim)
        edge_attr = edge_attr[:, :, :max_node, :]

        spatial_pos = reshape_features(spatial_pos, node_num)[:, :, :max_node]
        edge_attr_used = edge_attr
        for i in range(self.args.num_layers):
            x, _ = getattr(self, f"conv{i}")(x, spatial_pos=spatial_pos, edge_attr=edge_attr_used) 
            x = F.relu(x)

        x = x.permute(1, 0, 2)
        x = inverse_reshape_features(x, node_num)

        x = x.index_select(0, last_node_idx)
        
        x = self.linear4(x + noise_y.unsqueeze(-1))
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.linear(x)

        return x.squeeze()
