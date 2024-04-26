#%%
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
from torch_geometric import EdgeIndex

def preprocess(df, user_col, item_col, features, time, latent_dim=64):
    
    data = HeteroData()

    user_mapping = {idx: enum for enum, idx in enumerate(df[user_col].unique())}
    item_mapping = {idx: enum for enum, idx in enumerate(df[item_col].unique())}

    src = [user_mapping[idx] for idx in df[user_col]]
    dst = [item_mapping[idx] for idx in df[item_col]]
    edge_index = torch.tensor([src, dst])
  
    data['user'].x = torch.eye(len(user_mapping))
    data['item'].x = torch.randn(len(item_mapping), latent_dim)
    data['item'].num_nodes = len(item_mapping)
    
    time = torch.from_numpy(df[time].values).to(torch.long)
    
    for enum, feature in enumerate(features):
        feature_x = torch.from_numpy(df[feature].values).to(torch.long)
        data['user', feature, 'item'].edge_index = edge_index
        data['user', feature, 'item'].edge_label = feature_x
        data['user', feature, 'item'].time = time
        
    data = T.ToUndirected()(data)
        
    return data
 
def split(data, train_size=0.8):
    
    time = data['user', 'item'].time
    
    perm = time.argsort()
    train_index = perm[:int(train_size * perm.numel())]
    test_index = perm[int(train_size * perm.numel()):]
    
    return train_index, test_index 
    
def dataloader(data, train_index, test_index, device, batch_size=256, num_neighbors=[1], neg_sampling = 1):
    
    train_loader = LinkNeighborLoader(
        data=data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        neg_sampling_ratio = neg_sampling,
        edge_label_index=(('user', 'item'), data['user', 'item'].edge_index[:, train_index].to(device)),
        edge_label=data['user', 'item'].edge_label[train_index].to(device),
    )

    user_loader = NeighborLoader(
        data=data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes='user'
    )

    item_loader = NeighborLoader(
        data=data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes='item',
    )

    sparse_size = (data['user'].num_nodes, data['item'].num_nodes)

    test_edge_label_index = EdgeIndex(
        data['user', 'item'].edge_index[:, test_index].to(device),
        sparse_size=sparse_size,
    ).sort_by('row')[0]

    test_exclude_links = EdgeIndex(
        data['user', 'item'].edge_index[:, train_index].to(device),
        sparse_size=sparse_size,
    ).sort_by('row')[0]
    
    return train_loader, user_loader, item_loader, test_edge_label_index, test_exclude_links


class EarlyStopping:
    def __init__(self, k, patience=5, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score = 0
        self.k = k

    def __call__(self, val_score, model):
        score = val_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_score, model)
            self.counter = 0

    def save_checkpoint(self, val_score, model):
        if self.verbose:
            print(f'Validation score improved {self.val_score:.4f} --> {val_score:.4f} - Saving model...')
        torch.save(model.state_dict(), f'WLConvContinuous_{self.k}.pt')
        self.val_score = val_score

def seed_everything(seed):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True   
# %%
