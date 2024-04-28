#%%
import torch
from tqdm import tqdm
from torch_geometric.nn import GENConv
from torch_geometric.nn import to_hetero
from torch_geometric.nn.pool import MIPSKNNIndex
from torch_geometric.metrics import LinkPredPrecision
from torch_geometric.metrics import LinkPredRecall
from torch_geometric.metrics import LinkPredF1
from torch_geometric.metrics import LinkPredMAP
from torch_geometric.metrics import LinkPredNDCG

class Encoder(torch.nn.Module):
    
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = GENConv(in_channels = (-1, -1), out_channels = hidden_channels)
        self.conv2 = GENConv(in_channels = (-1, -1), out_channels = hidden_channels)
        self.conv3 = GENConv(in_channels = (-1, -1), out_channels = hidden_channels)

    def forward(self, batch_embedding_dict, batch_edge_index_dict):
            
        batch_embedding_dict = self.conv1(batch_embedding_dict, batch_edge_index_dict) 
        batch_embedding_dict = batch_embedding_dict.relu()
        batch_embedding_dict = self.conv2(batch_embedding_dict, batch_edge_index_dict) 
        batch_embedding_dict = batch_embedding_dict.relu()
        batch_embedding_dict = self.conv3(batch_embedding_dict, batch_edge_index_dict) 
        batch_embedding_dict = batch_embedding_dict.relu()
        
        return batch_embedding_dict
    
class Decoder(torch.nn.Module):
    
    def forward(self, batch_embedding_dict, edge_index):
        
        user_embedding = batch_embedding_dict['user'][edge_index[0]]
        item_embedding = batch_embedding_dict['item'][edge_index[1]]
        
        return (user_embedding * item_embedding).sum(dim=-1)
        
class RecommendationSystem(torch.nn.Module):
    
    def __init__(self, data, hidden_channels):
        super().__init__()
        self.encoder = Encoder(hidden_channels = hidden_channels)
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
