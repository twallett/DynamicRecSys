#%%
from utils import * 

plot_metric(data = 'MovieLens100k',
            metric='precision', 
            save=True)

plot_metric(data = 'MovieLens100k',
            metric='recall', 
            save=True)

plot_metric(data = 'MovieLens100k',
            metric='f1', 
            save=True)

plot_metric(data = 'MovieLens100k',
            metric='map', 
            save=True)

plot_metric(data = 'MovieLens100k',
            metric='ndcg', 
            save=True)

results_df = get_results(data = 'MovieLens100k',
                        metric='ndcg')

results_df

# %%
