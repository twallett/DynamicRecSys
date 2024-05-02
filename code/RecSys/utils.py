#%%

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def plot_metric(data, metric, save = False):
    
    K = [10,20,30,40,50,60,70,80,90,100]

    dir_init = os.getcwd()
    cdir = dir_init + os.sep + data + os.sep + 'results'
    os.chdir(cdir)
    dir = os.listdir()
    dir.sort()
    
    model_names = [name[:-12] for name in dir]

    colors = plt.cm.tab20.colors

    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'x', '+', '|', '_', '.']

    handles = []

    for idx, name in enumerate(model_names):
        path = cdir + os.sep + dir[idx]
        results_df = pd.read_csv(path, index_col=0)
        color = colors[idx % len(colors)]
        plt.plot(results_df.K, results_df[f"{metric}"], label=name, color=color)
        plt.scatter(results_df.K, results_df[f"{metric}"], color=color, marker=markers[idx])
        handles.append(mlines.Line2D([], [], color=color, marker=markers[idx], markersize=5, label=name))
    plt.title(f"MovieLens100k {metric} @ K")
    plt.ylabel(f"{metric}")
    plt.xlabel("K")
    plt.xticks(K)
    plt.legend(handles = handles, loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid()
    if save:
        os.chdir(dir_init)
        if not os.path.exists('plots'):
            os.makedirs('plots')
        plt.savefig(os.path.join('plots', f"{data}_{metric}.pdf"), bbox_inches='tight')
    plt.show()
    
def get_results(data, metric):
    dir_init = os.getcwd()
    cdir = dir_init + os.sep + data + os.sep + 'results'
    os.chdir(cdir)
    dir = os.listdir()
    dir.sort()
    
    model_names = [name[:-12] for name in dir]

    results = {}

    for idx, name in enumerate(model_names):
        path = cdir + os.sep + dir[idx]
        results_df = pd.read_csv(path, index_col=0)
        
        results_df.columns = pd.MultiIndex.from_product([[name], results_df.columns])
        
        results[name] = results_df

    results_df = pd.concat(results.values(), axis=1)

    results_df.index = pd.Index(range(10, 101, 10), name="K")
 
    return results_df.xs(f'{metric}', axis=1, level=1)
# %%
