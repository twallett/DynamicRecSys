#%%
import torch
from torch.nn import ReLU
from torch.nn import Sequential
from tqdm import tqdm
from torch_geometric.nn import SimpleConv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GraphConv
from torch_geometric.nn import ResGatedGraphConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import GINConv
from torch_geometric.nn import EdgeConv
from torch_geometric.nn import FeaStConv
from torch_geometric.nn import LEConv
from torch_geometric.nn import GENConv
from torch_geometric.nn import WLConvContinuous
from torch_geometric.nn import Linear
from torch_geometric.nn import to_hetero
from torch_geometric.nn.pool import MIPSKNNIndex
from torch_geometric.nn.aggr import SetTransformerAggregation
from torch_geometric.metrics import LinkPredPrecision
from torch_geometric.metrics import LinkPredRecall
from torch_geometric.metrics import LinkPredF1
from torch_geometric.metrics import LinkPredMAP
from torch_geometric.metrics import LinkPredNDCG

class EncoderWLConvContinuous(torch.nn.Module):
    
    def __init__(self, unique_user, hidden_channels):
        super().__init__()
        self.conv1 = WLConvContinuous()
        self.lin1 = Linear(-1, unique_user)
        self.conv2 = WLConvContinuous()
        self.lin2 = Linear(-1, hidden_channels)
        self.conv3 = WLConvContinuous()
        self.lin3 = Linear(-1, hidden_channels)

    def forward(self, batch_embedding_dict, batch_edge_index_dict):
        
        batch_embedding_dict = self.conv1(self.lin1(batch_embedding_dict), batch_edge_index_dict) 
        batch_embedding_dict = batch_embedding_dict.relu()
        batch_embedding_dict = self.lin2(self.conv1(batch_embedding_dict, batch_edge_index_dict))
        batch_embedding_dict = batch_embedding_dict.relu()
        batch_embedding_dict = self.lin3(self.conv1(batch_embedding_dict, batch_edge_index_dict))
        batch_embedding_dict = batch_embedding_dict.relu()
        
        return batch_embedding_dict

class EncoderGENConv(torch.nn.Module):
    
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = GENConv(in_channels = (-1, -1), out_channels = hidden_channels)
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = GENConv(in_channels = (-1, -1), out_channels = hidden_channels)
        self.lin2 = Linear(-1, hidden_channels)
        self.conv3 = GENConv(in_channels = (-1, -1), out_channels = hidden_channels)
        self.lin3 = Linear(-1, hidden_channels)

    def forward(self, batch_embedding_dict, batch_edge_index_dict):
            
        batch_embedding_dict = self.lin1(self.conv1(batch_embedding_dict, batch_edge_index_dict))
        batch_embedding_dict = batch_embedding_dict.relu()
        batch_embedding_dict = self.lin2(self.conv2(batch_embedding_dict, batch_edge_index_dict))
        batch_embedding_dict = batch_embedding_dict.relu()
        batch_embedding_dict = self.lin3(self.conv3(batch_embedding_dict, batch_edge_index_dict))
        batch_embedding_dict = batch_embedding_dict.relu()
        
        return batch_embedding_dict

class EncoderLEConv(torch.nn.Module):
    
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = LEConv(in_channels = (-1, -1), out_channels = hidden_channels)
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = LEConv(in_channels = (-1, -1), out_channels = hidden_channels)
        self.lin2 = Linear(-1, hidden_channels)
        self.conv3 = LEConv(in_channels = (-1, -1), out_channels = hidden_channels)
        self.lin3 = Linear(-1, hidden_channels)

    def forward(self, batch_embedding_dict, batch_edge_index_dict):
            
        batch_embedding_dict = self.lin1(self.conv1(batch_embedding_dict, batch_edge_index_dict))
        batch_embedding_dict = batch_embedding_dict.relu()
        batch_embedding_dict = self.lin2(self.conv2(batch_embedding_dict, batch_edge_index_dict))
        batch_embedding_dict = batch_embedding_dict.relu()
        batch_embedding_dict = self.lin3(self.conv3(batch_embedding_dict, batch_edge_index_dict))
        batch_embedding_dict = batch_embedding_dict.relu()
        
        return batch_embedding_dict

class EncoderFeaStConv(torch.nn.Module):
    
    def __init__(self, unique_user, hidden_channels, heads):
        super().__init__()
        self.conv1 = FeaStConv(in_channels = -1, out_channels = hidden_channels, add_self_loops=False, heads= heads)
        self.conv2 = FeaStConv(in_channels = -1, out_channels = hidden_channels, add_self_loops=False, heads= heads)
        self.lin1 = Linear(-1, unique_user)
        self.lin2 = Linear(-1, hidden_channels)
        self.lin3 = Linear(-1, hidden_channels)

    def forward(self, batch_embedding_dict, batch_edge_index_dict):
            
        batch_embedding_dict = self.conv1(self.lin1(batch_embedding_dict), batch_edge_index_dict) + self.lin2(batch_embedding_dict)
        batch_embedding_dict = batch_embedding_dict.relu()
        batch_embedding_dict = self.conv2(batch_embedding_dict, batch_edge_index_dict) + self.lin3(batch_embedding_dict)
        batch_embedding_dict = batch_embedding_dict.relu()
        return batch_embedding_dict

class EncoderEdgeConv(torch.nn.Module):
    
    def __init__(self, unique_user, hidden_layer_size):
        super().__init__()
        nn = Sequential(
            Linear(-1, hidden_layer_size),
            ReLU(),
            Linear(-1, hidden_layer_size),
            ReLU(),
        )
        self.conv1 = EdgeConv(nn = nn)
        self.lin1 = Linear(-1, unique_user)

    def forward(self, batch_embedding_dict, batch_edge_index_dict):
            
        batch_embedding_dict = self.conv1(self.lin1(batch_embedding_dict), batch_edge_index_dict) 
        batch_embedding_dict = batch_embedding_dict.relu()
        
        return batch_embedding_dict
    
class EncoderGINConv(torch.nn.Module):
    
    def __init__(self, unique_user, hidden_layer_size, eps):
        super().__init__()
        nn = Sequential(
            Linear(-1, hidden_layer_size),
            ReLU(),
            Linear(-1, hidden_layer_size),
            ReLU(),
            Linear(-1, hidden_layer_size),
            ReLU(),
        )
        self.conv1 = GINConv(nn = nn, eps = eps)
        self.lin1 = Linear(-1, unique_user)

    def forward(self, batch_embedding_dict, batch_edge_index_dict):
            
        batch_embedding_dict = self.conv1(self.lin1(batch_embedding_dict), batch_edge_index_dict) 
        batch_embedding_dict = batch_embedding_dict.relu()
        
        return batch_embedding_dict

class EncoderTransformerConv(torch.nn.Module):
    
    def __init__(self, hidden_channels, heads):
        super().__init__()
        self.conv1 = TransformerConv(in_channels=(-1, -1), out_channels=hidden_channels, heads=heads, concat=False)
        self.conv2 = TransformerConv(in_channels=(-1, -1), out_channels=hidden_channels, heads=heads, concat=False)
        self.conv3 = TransformerConv(in_channels=(-1, -1), out_channels=hidden_channels, heads=heads, concat=False)
    
    def forward(self, batch_embedding_dict, batch_edge_index_dict):

        batch_embedding_dict = self.conv1(batch_embedding_dict, batch_edge_index_dict) 
        batch_embedding_dict = batch_embedding_dict.relu()
        batch_embedding_dict = self.conv2(batch_embedding_dict, batch_edge_index_dict) 
        batch_embedding_dict = batch_embedding_dict.relu()
        batch_embedding_dict = self.conv3(batch_embedding_dict, batch_edge_index_dict)
        batch_embedding_dict = batch_embedding_dict.relu()
        
        return batch_embedding_dict

class EncoderGATv2Conv(torch.nn.Module):
    
    def __init__(self, hidden_channels, heads):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels=(-1, -1), out_channels=hidden_channels, add_self_loops=False, heads=heads, concat=False)
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = GATv2Conv(in_channels=(-1, -1), out_channels=hidden_channels, add_self_loops=False, heads=heads, concat=False)
        self.lin2 = Linear(-1, hidden_channels)
        self.conv3 = GATv2Conv(in_channels=(-1, -1), out_channels=hidden_channels, add_self_loops=False, heads=heads, concat=False)
        self.lin3 = Linear(-1, hidden_channels)
    
    def forward(self, batch_embedding_dict, batch_edge_index_dict):

        batch_embedding_dict = self.conv1(batch_embedding_dict, batch_edge_index_dict) + self.lin1(batch_embedding_dict)
        batch_embedding_dict = batch_embedding_dict.relu()
        batch_embedding_dict = self.conv2(batch_embedding_dict, batch_edge_index_dict) + self.lin2(batch_embedding_dict)
        batch_embedding_dict = batch_embedding_dict.relu()
        batch_embedding_dict = self.conv3(batch_embedding_dict, batch_edge_index_dict) + self.lin3(batch_embedding_dict)
        batch_embedding_dict = batch_embedding_dict.relu()
        
        return batch_embedding_dict

class EncoderGATConv(torch.nn.Module):
    
    def __init__(self, hidden_channels, heads):
        super().__init__()
        self.conv1 = GATConv(in_channels=(-1, -1), out_channels=hidden_channels, add_self_loops=False, heads=heads, concat=False)
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = GATConv(in_channels=(-1, -1), out_channels=hidden_channels, add_self_loops=False, heads=heads, concat=False)
        self.lin2 = Linear(-1, hidden_channels)
        self.conv3 = GATConv(in_channels=(-1, -1), out_channels=hidden_channels, add_self_loops=False, heads=heads, concat=False)
        self.lin3 = Linear(-1, hidden_channels)
    
    def forward(self, batch_embedding_dict, batch_edge_index_dict):

        batch_embedding_dict = self.conv1(batch_embedding_dict, batch_edge_index_dict) + self.lin1(batch_embedding_dict)
        batch_embedding_dict = batch_embedding_dict.relu()
        batch_embedding_dict = self.conv2(batch_embedding_dict, batch_edge_index_dict) + self.lin2(batch_embedding_dict)
        batch_embedding_dict = batch_embedding_dict.relu()
        batch_embedding_dict = self.conv3(batch_embedding_dict, batch_edge_index_dict) + self.lin3(batch_embedding_dict)
        batch_embedding_dict = batch_embedding_dict.relu()
        
        return batch_embedding_dict

class EncoderResGatedGraphConv(torch.nn.Module):
    
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = ResGatedGraphConv(in_channels = (-1, -1), out_channels = hidden_channels)
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = ResGatedGraphConv(in_channels = (-1, -1), out_channels = hidden_channels)
        self.lin2 = Linear(-1, hidden_channels)

    def forward(self, batch_embedding_dict, batch_edge_index_dict):
            
        batch_embedding_dict = self.lin1(self.conv1(batch_embedding_dict, batch_edge_index_dict))
        batch_embedding_dict = batch_embedding_dict.relu()
        batch_embedding_dict = self.lin2(self.conv2(batch_embedding_dict, batch_edge_index_dict))
        batch_embedding_dict = batch_embedding_dict.relu()

        return batch_embedding_dict

class EncoderGraphConv(torch.nn.Module):
    
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = GraphConv(in_channels = (-1, -1), out_channels = hidden_channels)
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = GraphConv(in_channels = (-1, -1), out_channels = hidden_channels)
        self.lin2 = Linear(-1, hidden_channels)

    def forward(self, batch_embedding_dict, batch_edge_index_dict):
            
        batch_embedding_dict = self.lin1(self.conv1(batch_embedding_dict, batch_edge_index_dict))
        batch_embedding_dict = batch_embedding_dict.relu()
        batch_embedding_dict = self.lin2(self.conv2(batch_embedding_dict, batch_edge_index_dict))
        batch_embedding_dict = batch_embedding_dict.relu()
  
        return batch_embedding_dict

class EncoderSAGEConv(torch.nn.Module):
    
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels = (-1, -1), out_channels = hidden_channels)
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = SAGEConv(in_channels = (-1, -1), out_channels = hidden_channels)
        self.lin2 = Linear(-1, hidden_channels)

    def forward(self, batch_embedding_dict, batch_edge_index_dict):
            
        batch_embedding_dict = self.lin1(self.conv1(batch_embedding_dict, batch_edge_index_dict))
        batch_embedding_dict = batch_embedding_dict.relu()
        batch_embedding_dict = self.lin2(self.conv2(batch_embedding_dict, batch_edge_index_dict))
        batch_embedding_dict = batch_embedding_dict.relu()
        
        return batch_embedding_dict

class EncoderSimpleConv(torch.nn.Module):
    
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SimpleConv(aggr = SetTransformerAggregation(hidden_channels))
        self.lin1 = Linear(-1, hidden_channels)

    def forward(self, batch_embedding_dict, batch_edge_index_dict):
        
        batch_embedding_dict = self.conv1(self.lin1(batch_embedding_dict), batch_edge_index_dict) 
        batch_embedding_dict = batch_embedding_dict.relu()

        return batch_embedding_dict
    
class Decoder(torch.nn.Module):
    
    def forward(self, batch_embedding_dict, edge_index):
        
        user_embedding = batch_embedding_dict['user'][edge_index[0]]
        item_embedding = batch_embedding_dict['item'][edge_index[1]]
        
        return (user_embedding * item_embedding).sum(dim=-1)
        
class RecommendationSystem(torch.nn.Module):
    
    def __init__(self, data, unique_user, hidden_layer_size, model, hidden_channels, heads, epsilon):
        super().__init__()
        if model == 'SimpleConv':
            self.encoder = EncoderSimpleConv(hidden_channels=hidden_channels)
        elif model == 'SAGEConv':
            self.encoder = EncoderSAGEConv(hidden_channels=hidden_channels)
        elif model == 'GraphConv':
            self.encoder = EncoderGraphConv(hidden_channels=hidden_channels)
        elif model == 'ResGatedGraphConv':
            self.encoder = EncoderResGatedGraphConv(hidden_channels=hidden_channels)
        elif model == 'GATConv':
            self.encoder = EncoderGATConv(hidden_channels=hidden_channels,heads=heads)
        elif model == 'GATv2Conv':
            self.encoder = EncoderGATv2Conv(hidden_channels=hidden_channels,heads=heads)
        elif model == 'TransformerConv':
            self.encoder = EncoderTransformerConv(hidden_channels=hidden_channels,heads=heads)
        elif model == 'GINConv':
            self.encoder = EncoderGINConv(unique_user=unique_user, hidden_layer_size=hidden_layer_size, eps = epsilon)
        elif model == 'EdgeConv':
            self.encoder = EncoderEdgeConv(unique_user=unique_user, hidden_layer_size=hidden_layer_size)
        elif model == 'FeaStConv':
            self.encoder = EncoderFeaStConv(unique_user=unique_user, hidden_channels=hidden_channels,heads=heads)
        elif model == 'LEConv':
            self.encoder = EncoderLEConv(hidden_channels=hidden_channels)
        elif model == 'GENConv':
            self.encoder = EncoderGENConv(hidden_channels=hidden_channels)
        elif model == 'WLConvContinuous':
            self.encoder = EncoderWLConvContinuous(unique_user=unique_user, hidden_channels=hidden_channels)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr = 'sum')
        self.decoder = Decoder()
    
    def forward(self, batch_embedding_dict, batch_edge_index_dict, edge_index):
        batch_embedding_dict = self.encoder(batch_embedding_dict, batch_edge_index_dict)
        return self.decoder(batch_embedding_dict, edge_index)
    
def train(model, optimizer, train_loader, device):      
      
    model.train()
    total_loss = total_examples = 0
    for batch in tqdm(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch.x_dict, 
                    batch.edge_index_dict,
                    batch['user','item'].edge_label_index)
        y = batch['user', 'item'].edge_label

        loss = torch.nn.functional.mse_loss(out, y.float())
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * y.numel()
        total_examples += y.numel()
         
    return total_loss / total_examples

@torch.no_grad()
def test(model, item_loader, user_loader, test_edge_label_index, test_exclude_links, K, device):
    model.eval()
    positive_item_embeddings = []
    for batch in item_loader:
        batch = batch.to(device)
        batch_embedding_item = model.encoder(batch.x_dict,
                                             batch.edge_index_dict)['item']
        batch_embedding_item_positive = batch_embedding_item[:batch['item'].batch_size]
        
        positive_item_embeddings.append(batch_embedding_item_positive)
    horizontal_stack_positive_item_embeddings = torch.cat(positive_item_embeddings,dim = 0)
    del positive_item_embeddings
    
    max_inner_product_search = MIPSKNNIndex(horizontal_stack_positive_item_embeddings)

    precision = LinkPredPrecision(k=K).to(device)
    recall = LinkPredRecall(k=K).to(device)
    f1 = LinkPredF1(k=K).to(device)
    map = LinkPredMAP(k=K).to(device)
    ndcg = LinkPredNDCG(k=K).to(device)
    
    number_processed = 0
    for batch in user_loader:
        batch = batch.to(device)
        batch_embedding_user = model.encoder(batch.x_dict, 
                                             batch.edge_index_dict)['user']
        batch_embedding_user_positive = batch_embedding_user[:batch['user'].batch_size]
        
        edge_index = test_edge_label_index.sparse_narrow(
            dim=0,
            start=number_processed,
            length=batch_embedding_user_positive.size(0))
        
        exclude_links = test_exclude_links.sparse_narrow(
            dim=0,
            start=number_processed,
            length=batch_embedding_user_positive.size(0))
        
        number_processed += batch_embedding_item_positive.size(0)

        _, pred_index_mat = max_inner_product_search.search(batch_embedding_user_positive, K, exclude_links)

        precision.update(pred_index_mat, edge_index)
        recall.update(pred_index_mat, edge_index)
        f1.update(pred_index_mat, edge_index)
        map.update(pred_index_mat, edge_index)
        ndcg.update(pred_index_mat, edge_index)
    
    precision = precision.compute()
    recall = recall.compute()
    f1 = f1.compute()
    map = map.compute()
    ndcg = ndcg.compute()
    return precision, recall, f1, map, ndcg
