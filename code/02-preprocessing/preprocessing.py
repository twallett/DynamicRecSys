#%%

import pandas as pd 

df = pd.read_csv("movie_data.csv",
                 index_col=0)

#%%

from sklearn.preprocessing import LabelEncoder

features = ["age", "gender", "occupation", "zipcode"]

df_features = df[features]

df_numerical_features = df_features.select_dtypes(exclude=["object_","flexible"])
df_categorical_features = df_features.select_dtypes(exclude=["number", "bool_"])

encoder = LabelEncoder()
df_categorical_features = df_categorical_features.apply(encoder.fit_transform)

df_numerical_features = df_numerical_features.merge(df_categorical_features,
                                                    on = df_numerical_features.index)

features = df_numerical_features.iloc[:,1:].to_numpy()

# print statement here 

#%%

import numpy as np
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
df["item_idx"] = encoder.fit_transform(df.item_id)

n_arms = len(df.item_id.unique())

unique_items = df.item_id.unique()
unique_items.sort()

data = np.zeros( (len(df), n_arms) )

for i in range( len(data) ):
    data[i,df.item_idx[i]] = df.rating[i]

#%%

# lets test the model 

import matplotlib.pyplot as plt
from model import LinUCB

model = LinUCB(n_arms = n_arms,
               n_features = features.shape[1],
               random_seed = 123)

recs, rewards, matrix = model.train(data[:10000,:], features[:10000,:])

plt.plot(rewards)
plt.show()

# %%
