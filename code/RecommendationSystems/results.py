#%%
from utils import * 

plot_metric(metric='precision', 
            save=True)

plot_metric(metric='recall', 
            save=True)

plot_metric(metric='f1', 
            save=True)

plot_metric(metric='map', 
            save=True)

plot_metric(metric='ndcg', 
            save=True)

recall_df = get_results(metric='ndcg')

recall_df
# %%

#pip install git+https://github.com/pyg-team/pyg-lib.git

# %%
