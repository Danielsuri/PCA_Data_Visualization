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
                           , columns=['PC1', 'PC2'])

finalDf = pd.concat([principalDf, featDf], axis=1)
finalDf.rename(columns={0: 'target'}, inplace=True)
# delete unused vars
del (idx, items, all_files, all_pca_data)

explained_variance = pca.explained_variance_ratio_

# plot:

targets = list(set(list(finalDf['target'])))
targets.sort()
fig, ax = plt.subplots(figsize=(6, 5))
for items in targets:
    x = finalDf.loc[finalDf['target'] == items, 'PC1']
    y = finalDf.loc[finalDf['target'] == items, 'PC2']
    ax.scatter(x, y, label=items)
    # uncomment for annotate:
    # for dot in enumerate(x):
    #     plt.annotate(items, xy=(x.values[dot[0]], y.values[dot[0]]), xytext=(10, 10),
    #                  textcoords="offset points", arrowprops=dict(arrowstyle="->"))

ax.legend()
ax.set_xlabel('PC1', fontsize=15)
ax.set_ylabel('PC2', fontsize=15)
ax.set_title('2 Component PCA', fontsize=20)
ax.grid()


plt.show()
