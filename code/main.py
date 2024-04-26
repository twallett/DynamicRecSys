#%%
import torch
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.nn import GATConv
from torch_geometric.nn import Linear
from torch_geometric.nn import to_hetero
from torch_geometric.nn.pool import MIPSKNNIndex
from torch_geometric.metrics import LinkPredRecall
from utils import *

############### Hyperparameters ###############

SEED = 123
SPLIT = 7/8
BATCH_SIZE = 1
NEIGHBORS = [1]
NEG_SAMPLING = 1

EPOCHS = 1500
LR = 1e-02
LATENT_DIM = 12
HEADS = 2
K = 1

############### Hyperparameters ###############

seed_everything(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = [[0,0,5,1],
        [0,1,1,2],
        [0,2,1,3],
        [1,0,1,4],
        [1,1,5,5],
        [1,2,5,6],
        [2,2,5,7],
        [2,1,5,8]]

df = pd.DataFrame(data, columns = ["user_id", "item_id", "rating", "timestamp"])

data = preprocess(df = df,
                  user_col = 'user_id',
                  item_col = 'item_id',
                  features = ['rating'],
                  time = 'timestamp',
                  latent_dim = LATENT_DIM)

train_index, test_index = split(data = data, 
                                train_size = SPLIT)

train_loader, user_loader, item_loader, test_edge_label_index, test_exclude_links = dataloader(data, 
                                                                                               train_index, 
                                                                                               test_index, 
                                                                                               device, 
                                                                                               batch_size=BATCH_SIZE)

class Encoder(torch.nn.Module):
    
    def __init__(self, hidden_channels, heads):
        super().__init__()
        self.conv1 = GATConv(in_channels = (-1, -1), out_channels = hidden_channels, add_self_loops= False, heads = heads, concat=False)
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = GATConv(in_channels = (-1, -1), out_channels = hidden_channels, add_self_loops= False, heads = heads, concat=False)
        self.lin2 = Linear(-1, hidden_channels)
        self.conv3 = GATConv(in_channels = (-1, -1), out_channels = hidden_channels, add_self_loops= False, heads = heads, concat=False)
        self.lin3 = Linear(-1, hidden_channels)
    
    def forward(self, batch_embedding_dict, batch_edge_index_dict):
            
        batch_embedding_dict = self.conv1(batch_embedding_dict, batch_edge_index_dict) + self.lin1(batch_embedding_dict)
        batch_embedding_dict = batch_embedding_dict.relu()
        batch_embedding_dict = self.conv2(batch_embedding_dict, batch_edge_index_dict) + self.lin2(batch_embedding_dict)
        batch_embedding_dict = batch_embedding_dict.relu()
        batch_embedding_dict = self.conv3(batch_embedding_dict, batch_edge_index_dict) + self.lin3(batch_embedding_dict)
        batch_embedding_dict = batch_embedding_dict.relu()
        
        return batch_embedding_dict
    
class Decoder(torch.nn.Module):
    
    def forward(self, batch_embedding_dict, edge_index):
        
        user_embedding = batch_embedding_dict['user'][edge_index[0]]
        item_embedding = batch_embedding_dict['item'][edge_index[1]]
        
        return (user_embedding * item_embedding).sum(dim=-1)
        
class RecommendationSystem(torch.nn.Module):
    
    def __init__(self, data, hidden_channels, heads):
        super().__init__()
        self.encoder = Encoder(hidden_channels = hidden_channels, heads=heads)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr = 'sum')
        self.decoder = Decoder()
    
    def forward(self, batch_embedding_dict, batch_edge_index_dict, edge_index):
        batch_embedding_dict = self.encoder(batch_embedding_dict, batch_edge_index_dict)
        return self.decoder(batch_embedding_dict, edge_index)
        
model = RecommendationSystem(data = data, 
                             hidden_channels = LATENT_DIM,
                             heads = HEADS).to(device)

optimizer = torch.optim.Adam(model.parameters(), 
                             lr=LR)

aloss = []
metrics = []
for _ in range(EPOCHS):
    
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
        
    print(total_loss / total_examples)
    aloss.append(total_loss / total_examples)
    
    model.eval()
    positive_item_embeddings = []
    for batch in item_loader:
        batch = batch.to(device)
        batch_embedding_item = model.encoder(batch.x_dict,
                                             batch.edge_index_dict)['item']
        batch_embedding_item_positive = batch_embedding_item[:batch['item'].batch_size]
        
        positive_item_embeddings.append(batch_embedding_item_positive)
    horizontal_stack_positive_item_embeddings = torch.cat(positive_item_embeddings, dim = 0)
    del positive_item_embeddings
    
    max_inner_product_search = MIPSKNNIndex(horizontal_stack_positive_item_embeddings)

    recall = LinkPredRecall(k=K).to(device)
    
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
                
        recall.update(pred_index_mat, edge_index)

    recall = recall.compute()
    print(f"recall @ {K}: {recall}")
    metrics.append(recall)

#%%

plt.plot(aloss)
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("MSE Loss: Case Study")
plt.savefig('CaseStudyLoss.pdf')
plt.show()

metrics = torch.tensor(metrics)
plt.plot(metrics)
plt.xlabel("Epochs")
plt.ylabel("Recall")
plt.title("Recall @ 1: Case Study")
plt.savefig('CaseStudyRecall.pdf')
plt.show()

#%%
