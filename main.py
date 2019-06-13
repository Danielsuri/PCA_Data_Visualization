import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import glob

# loading dataset into Pandas DataFrame

all_files = glob.glob('PCA/*.txt')
all_files.sort()
all_pca_data = pd.DataFrame()

for filename in all_files:
    data_frame = pd.read_csv(filename, sep='\t', skiprows=13, names=['Wave_Length', filename[4::]])
    all_pca_data[filename[4::]] = data_frame[filename[4::]]
all_pca_data.insert(loc=0, column='Wave_Length', value=data_frame['Wave_Length'])
all_pca_data_T = all_pca_data.T
# set first row as columns names:
all_pca_data_T.columns = all_pca_data_T.iloc[0]

features = list(all_pca_data_T.index.values)
# del features[0]
for idx, items in enumerate(features[1::]):
    features[idx + 1] = items.split('_')[0]
featDf = pd.DataFrame([features])
featDf.drop(columns=0, inplace=True)
featDf = featDf.T
featDf.reset_index(inplace=True)
featDf.drop(columns='index', inplace=True)
x = all_pca_data_T.iloc[1:].values
y = all_pca_data_T.iloc[1:].index.values
x = StandardScaler().fit_transform(x)
df = pd.DataFrame(data=x, index=features[1::])

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data=principalComponents
                           , columns=['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, featDf], axis=1)
finalDf.rename(columns={0: 'target'}, inplace=True)
del (idx, items, all_files)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_title('2 Component PCA', fontsize=20)
targets = ['Air', 'Egg', 'Yogurt']
colors = ['r', 'g', 'b']
for target, color in zip(targets, colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c=color
               , s=50)
ax.legend(targets)
ax.grid()

