#%%

import pandas as pd
from utils import *
from model import RecommendationSystem, train, test

############### Hyperparameters ###############

SEED = 123
SPLIT = 0.8
BATCH_SIZE = 1024
NEIGHBORS = [256,128]
NEG_SAMPLING = 1
PATIENCE = 20

EPOCHS = 1000
LR = 3e-03
LATENT_DIM = 64
HEADS = 8
K = [10,20,30,40,50,60,70,80,90,100]

############### Hyperparameters ###############

seed_everything(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv("MovieLens100k.csv",
                 index_col=0)

data = preprocess(df,
                  user_col='user_id',
                  item_col='item_id',
                  features=['rating'],
                  time='timestamp',
                  latent_dim= LATENT_DIM)

train_index, test_index = split(data,
                                train_size=SPLIT)

train_loader, user_loader, item_loader, test_edge_label_index, test_exclude_links = dataloader(data,
                                                                                               train_index = train_index,
                                                                                               test_index = test_index,
                                                                                               device=device,
                                                                                               batch_size=BATCH_SIZE,
                                                                                               num_neighbors=NEIGHBORS,
                                                                                               neg_sampling=NEG_SAMPLING)

model = RecommendationSystem(data = data, 
                             hidden_channels = LATENT_DIM,
                             heads=HEADS).to(device)

optimizer = torch.optim.SGD(model.parameters(), 
                             lr= LR)

best_precision_scores = {k: 0 for k in K}
best_recall_scores = {k: 0 for k in K}
best_f1_scores = {k: 0 for k in K}
best_map_scores = {k: 0 for k in K}
best_ndcg_scores = {k: 0 for k in K}

for k in K:
    early_stopping = EarlyStopping(patience=PATIENCE,
                                   verbose=True,
                                   k = k)
    best_precision = 0
    best_recall = 0
    best_f1 = 0
    best_map = 0
    best_ndcg = 0
    for epoch in range(EPOCHS):
        print(f"epoch {epoch}")
        train_loss = train(model=model, 
                        optimizer=optimizer,
                        train_loader=train_loader,
                        device=device)
        print(f"loss {train_loss.__round__(4)}")
        
        precision, recall, f1, map, ndcg = test(model=model,
                                                item_loader=item_loader,
                                                user_loader=user_loader,
                                                test_edge_label_index= test_edge_label_index,
                                                test_exclude_links= test_exclude_links,
                                                K = k,
                                                device=device)
        print(f"recall @ {k} {recall.item().__round__(4)}")
        
        if precision.item() > best_precision:
            best_precision = precision.item().__round__(4)
            best_precision_scores[k] = best_precision
       
        if recall.item() > best_recall:
            best_recall = recall.item().__round__(4)
            best_recall_scores[k] = best_recall
            
        if f1.item() > best_f1:
            best_f1 = f1.item().__round__(4)
            best_f1_scores[k] = best_f1

        if map.item() > best_map:
            best_map = map.item().__round__(4)
            best_map_scores[k] = best_map

        if ndcg.item() > best_ndcg:
            best_ndcg = ndcg.item().__round__(4)
            best_ndcg_scores[k] = best_ndcg
            
        early_stopping(recall.item(), model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
results_df = pd.DataFrame({
    'K': K,
    'precision': list(best_precision_scores.values()),
    'recall': list(best_recall_scores.values()),
    'f1': list(best_f1_scores.values()),
    'map': list(best_map_scores.values()),
    'ndcg': list(best_ndcg_scores.values())
})

results_df.to_csv("GATConv_results.csv")

#%%
# !pip install torch_geometric

# import torch

# !pip uninstall torch-scatter torch-sparse torch-geometric torch-cluster  --y
# !pip install torch-scatter -f https://data.pyg.org/whl/torch-{torch.__version__}.html
# !pip install torch-sparse -f https://data.pyg.org/whl/torch-{torch.__version__}.html
# !pip install torch-cluster -f https://data.pyg.org/whl/torch-{torch.__version__}.html
# !pip install git+https://github.com/pyg-team/pytorch_geometric.git

# !pip install faiss-gpu

# %%
