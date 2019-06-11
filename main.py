import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import glob

# %matplotlib inline
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# loading dataset into Pandas DataFrame

all_files = glob.glob('PCA/*.txt')
all_files.sort()
all_pca_data = pd.DataFrame()

for filename in all_files:
    data_frame = pd.read_csv(filename, sep='\t', skiprows=13, names=['Wave_Length', filename[4::]])
    all_pca_data[filename[4::]] = data_frame[filename[4::]]
all_pca_data.insert(loc=0, column='Wave_Length', value=data_frame['Wave_Length'])

headers = list(all_pca_data.columns)
del headers[0]

y = list(all_pca_data.columns)
del y[0]
for idx, items in enumerate(y):
    y[idx] = items.split('_')[0]

x = all_pca_data.loc[:, headers].values
x = StandardScaler().fit_transform(x)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data=principalComponents
                           , columns=['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, y], axis=1)
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(1, 1, 1)
# ax.set_xlabel('Principal Component 1', fontsize=15)
# ax.set_ylabel('Principal Component 2', fontsize=15)
# ax.set_title('2 Component PCA', fontsize=20)
# targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
# colors = ['r', 'g', 'b']
# for target, color in zip(targets, colors):
#     indicesToKeep = finalDf['target'] == target
#     ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
#                , finalDf.loc[indicesToKeep, 'principal component 2']
#                , c=color
#                , s=50)
# ax.legend(targets)
# ax.grid()

# #########################################
df = pd.read_csv(url
                 , names=['sepal length', 'sepal width', 'petal length', 'petal width', 'target'])
print(df)
features = ['sepal length', 'sepal width', 'petal length', 'petal width']
x = df.loc[:, features].values
y = df.loc[:, ['target']].values
x = StandardScaler().fit_transform(x)
pd.DataFrame(data=x, columns=features).head()
# PCA Projection to 2D
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data=principalComponents
                           , columns=['principal component 1', 'principal component 2'])
principalDf.head(5)
df[['target']].head()
finalDf = pd.concat([principalDf, df[['target']]], axis=1)
finalDf.head(5)
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_title('2 Component PCA', fontsize=20)

targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets, colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c=color
               , s=50)
ax.legend(targets)
ax.grid()
pca.explained_variance_ratio_
