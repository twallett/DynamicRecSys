#%%

import pandas as pd

data_column_names = ['user_id','item_id','rating','timestamp']
data_df = pd.read_csv("u.data.txt",
                      sep='\t',
                      names=data_column_names)
data_df = data_df[data_df.rating >= 4]
data_df.sort_values('timestamp', inplace=True)

print(f"The user-item interactions dataset:", '\n' '\n', data_df, '\n')
print(f"Total user-item of interactions: {len(data_df)}", '\n')
print(f"The amount of unique users: {len(data_df.user_id.unique())}", '\n')
print(f"The amount of unique items: {len(data_df.item_id.unique())}")

#%%

user_column_names = ['user_id','age','gender','occupation', 'zipcode']
user_df = pd.read_csv('u.user.txt',
                      sep='|',
                      names=user_column_names)

print(f"The user demographics dataset:", '\n' '\n', user_df, '\n')

#%%

# Merging both on to 'data_df' based on column 'user_id'

data_df = data_df.merge(user_df,
                        on = 'user_id')

print(f"The merged dataset:", '\n' '\n', data_df, '\n')

#%%

import os 

os.chdir("/Users/tylerwallett/Documents/Documents/GitHub/GNN/code/02-preprocessing")

data_df.to_csv('movie_data.csv')

#%%
