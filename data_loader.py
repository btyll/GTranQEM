import os
import torch
import pickle
from torchpack.datasets.dataset import Dataset
from torch.utils.data import DataLoader
from torch_geometric.data import Data

def batch_graph_data(data, args):
    x = torch.tensor([])
    edge_index = torch.tensor([])
    batch = torch.tensor([])
    num_nodes_tensor = torch.tensor([])
    edge_attr = torch.tensor([])
    pe = torch.tensor([])
    accu_num_nodes = 0
    edge_attr_dim = 6
    max_num_nodes = 500 
    spatial_pos = torch.tensor([])
    for b in range(data['x'].shape[0]):
        num_nodes = len(data['node_idx'])
        node_feature = data['x'][b]
        x = torch.cat((x, node_feature[:num_nodes,:]))
        pe = torch.cat((pe, torch.tensor(data['pe_feat'])))
        assert pe.shape == x.shape, f'{pe.shape}, {x.shape}'
        x = torch.cat((x, pe), dim=1)
         
        spatial_pos_i = data['shortest_path'][b].clone().detach()

        num_nodes_tensor = torch.cat((num_nodes_tensor, torch.tensor([num_nodes])))

        x = x[:num_nodes]
        spatial_pos_i = spatial_pos_i[:num_nodes, :num_nodes]
        spatial_pos = torch.cat((spatial_pos, spatial_pos_i))
        adj = data['adj'][b][:num_nodes, :num_nodes]
        assert spatial_pos.shape[0] == x.shape[0], f'{spatial_pos.shape}, {x.shape}'

        edge_index_a, edge_index_b = torch.where(adj==1)
        assert num_nodes > max(edge_index_a) and num_nodes > max(edge_index_b)
        edge_index = torch.cat((edge_index, accu_num_nodes + torch.vstack((edge_index_a, edge_index_b))),1)

        edge_attr_per_graph = torch.zeros((num_nodes, max_num_nodes, edge_attr_dim))
        for i, j in zip(edge_index_a, edge_index_b):
            edge_attr_per_graph[int(i)][int(j)] = torch.tensor(data['edge_feature'][(int(i), int(j))]).reshape(-1)
        edge_attr = torch.cat((edge_attr, edge_attr_per_graph))
        
        batch = torch.cat((batch, torch.full((num_nodes,), b)))

        accu_num_nodes += num_nodes
        assert x.shape[0] == accu_num_nodes

    tensor_to_cat = torch.zeros((spatial_pos.shape[0], max_num_nodes - spatial_pos.shape[1]))

    assert x.shape[0] == num_nodes_tensor.sum() == spatial_pos.shape[0]
    out_data = Data(
        x = x.to(torch.float32),
        edge_index = edge_index.to(torch.int64),
        edge_attr = edge_attr.to(torch.float32),
        y = data['y'].real.to(torch.float32),
        num_nodes_tensor = num_nodes_tensor.to(torch.long),
        batch = batch,
        noise_y = data['noise_y'].real.to(torch.float32),
        cir_id = data['cir_id'][0],
        spatial_pos = torch.cat([spatial_pos, tensor_to_cat], dim=1).to(torch.int64)
    )
    return out_data

class QEMDataset(Dataset):
    def __init__(self, root_dir, qubits):
        self.root_dir = root_dir
        self.num_features = 14 + qubits
                
    def __len__(self):
        return int(self.root_dir.split('_')[-1]) 

    def __getitem__(self, index):
        data = pickle.load(open(self.root_dir + '/data_' +str(index) + '.pkl', 'rb'))
        data['cir_id'] = str(index)
        return data


import torch
from torch.utils.data import Subset, random_split

def split_dataset_extrapolation(dataset: QEMDataset, data_path, total_num=400):
    test_indices = [i for i in range(total_num) if dataset.__getitem__(i)['trotter_step'] > 15]
    
    train_val_indices = [i for i in range(total_num) if dataset.__getitem__(i)['trotter_step'] <= 15]
    
    train_size = int(0.85 * len(train_val_indices))
    val_size = len(train_val_indices) - train_size
    
    train_indices, val_indices = random_split(train_val_indices, [train_size, val_size])
    
    print('train/val/test', len(train_indices), len(val_indices), len(test_indices))

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)

    index_dict =  {
        'train_indices': train_indices,
        'val_indices': val_indices,
        'test_indices': test_indices
    }
    os.makedirs(f'{data_path}/splits', exist_ok=True)
    pickle.dump(index_dict, open(f'{data_path}/splits/split_indices.pkl', 'wb'))
    
    return train_set, val_set, test_set


def load_ev_dataset(dataset_path, num_qubits, device, batch_size=1, test_type='intra', args=None, train_size=750, valid_size=150, test_size=300):
    train_dataset_list = []
    test_dataset_list = []
    valid_dataset_list = []
    mydataset = QEMDataset(dataset_path, num_qubits)

    if test_type == 'intra':
        train_dataset = torch.utils.data.Subset(mydataset, range(train_size))
        valid_dataset = torch.utils.data.Subset(mydataset, range(train_size, train_size + valid_size))
        test_dataset = torch.utils.data.Subset(mydataset, range(train_size + valid_size, train_size + valid_size + test_size))
    else:
        print('generating dataset for extrapolation data settings')
        train_dataset, valid_dataset, test_dataset = split_dataset_extrapolation(mydataset, dataset_path, total_num=train_size + valid_size + test_size)
    
    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0)
    valid_loader = DataLoader(dataset=valid_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0)
    test_loader =  DataLoader(dataset=test_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=0)

    batch_func = batch_graph_data
    print("Processing train batches:")
    for idx, data in enumerate(train_loader):
        print(f"\rTrain batch: {idx+1}/{len(train_loader)}", end="", flush=True)
        train_dataset_list.append(batch_func(data, args).to(device))
    print("\nProcessing valid batches:")
    for idx, data in enumerate(valid_loader):
        print(f"\rValid batch: {idx+1}/{len(valid_loader)}", end="", flush=True)
        valid_dataset_list.append(batch_func(data, args).to(device))
    print("\nProcessing test batches:")
    for idx, data in enumerate(test_loader):
        print(f"\rTest batch: {idx+1}/{len(test_loader)}", end="", flush=True)
        test_dataset_list.append(batch_func(data, args).to(device))
    print()
    num_features = mydataset.num_features
    return mydataset, train_dataset_list, valid_dataset_list, test_dataset_list, num_features

