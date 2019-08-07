import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import glob
from matplotlib.font_manager import FontProperties
from numpy.random import rand
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.image import AxesImage

all_files = glob.glob('plot/*.txt')
all_files.sort()

df = pd.DataFrame()
for file_name in all_files:
    col_name = file_name.split("\\")[1].split('.')[0]
    data = pd.read_csv(file_name, sep='\t', skiprows=13, names=['Wave_Length', col_name])
    df.insert(loc=0, column=col_name, value=data[col_name].values)

df = df.T
df.columns = data['Wave_Length']
df = df.T
# Get dark values and subtract them from all data
dark = df['Dark_NQ51A06081_21-20-59-000'].values
df.drop(columns='Dark_NQ51A06081_21-20-59-000', inplace=True)
df_minus_dark = df.sub(dark, axis='index')

clean_sam = []
str_to_find = 'clean'

sample_names = df.columns.values
for sample in sample_names:
    if str_to_find in sample:
        clean_sam.append(sample)
# Mean over all clean samples
mean_clean_sam = df[clean_sam].mean(1)
# Remove all clean samples to stay only with measurements data
df_minus_dark.drop(columns=clean_sam, inplace=True)

air_sam =[]
str_to_find = 'Air'
for sample in sample_names:
    if str_to_find in sample:
        air_sam.append(sample)
# Mean over all clean samples
mean_air_sam = df[clean_sam].mean(1)
# Remove all air samples to stay only with measurements data
df_minus_dark.drop(columns=air_sam, inplace=True)

sams =[]
str_to_find = 'sample'
for sample in sample_names:
    if str_to_find in sample:
        sams.append(sample)
sams_df = df_minus_dark[sams]
materials_df = df_minus_dark.drop(columns=sams)
sams_df = sams_df.div(mean_clean_sam, axis=0)
materials_df = materials_df.div(mean_air_sam, axis=0)

# Divide by air mean => Absorbance dataframe
# absorbance_df = df_minus_dark.div(mean_clean_sam, axis=0)

trans_df = pd.concat([sams_df, materials_df], axis=1, sort=False)

gg = -1 * (np.log10(trans_df.values))
ab = np.divide(gg, abs(gg[346]))
abs_df = pd.DataFrame(index=trans_df.index.values, columns=trans_df.columns.values, data=ab)

abs_df = abs_df.T

for names in abs_df.index.values:
    abs_df.rename(index={names: names.split('_')[0]}, inplace=True)

x = StandardScaler().fit_transform(abs_df.iloc[1:].values)
y = abs_df.iloc[1:].index.values
waves = abs_df.columns.values
pca = PCA()
principalComponents = pca.fit_transform(x)
loadings = pca.fit(x).components_
explained_variance = pca.explained_variance_ratio_
pca_col_names = []
for idx, items in enumerate(explained_variance):
    pca_col_names.append('PC' + str((idx + 1)))

all_principalDf = pd.DataFrame(data=principalComponents
                               , columns=pca_col_names)

principalDf = all_principalDf[['PC1', 'PC2']]

principalDf.insert(loc=0, column='targets', value=y)

loadingsDf = pd.DataFrame(data=pca.components_.T * np.sqrt(pca.explained_variance_),
                          columns=pca_col_names)


# plot:

targets = list(set(list(principalDf['targets'])))
targets.sort()
fig, ax = plt.subplots(figsize=(10, 5))
for items in targets:
    x = principalDf.loc[principalDf['targets'] == items, 'PC1'].values
    y = principalDf.loc[principalDf['targets'] == items, 'PC2'].values
    line, = ax.plot(x, y, 'o', label=items)

fontP = FontProperties()
fontP.set_size('small')
ax.legend(prop=fontP)
ax.set_xlabel('PC1   ' + '{0:.0%}'.format(explained_variance[0]))
ax.set_ylabel('PC2   ' + '{0:.0%}'.format(explained_variance[1]))
ax.set_title('Two Component PCA of all samples and materials')
ax.grid()
# plt.show()

# Loadings plot
fig2, ax2 = plt.subplots(figsize=(10, 5))
for idx, items in enumerate(loadingsDf.columns.values[:4]):
    ax2.plot(waves, loadings[idx], label=items + ' {:.4%}'.format(explained_variance[idx]))
ax2.set_xlabel('Wave Length')
ax2.set_ylabel('Loadings')
ax2.set_title('Loadings')
ax2.legend(prop=fontP)
ax2.grid()



fig3, ax3 = plt.subplots(figsize=(10, 5))
ax3.plot(np.arange(1, len(explained_variance[:4]) + 1), explained_variance[:4] * 100, marker='o')
for idx, txt in enumerate(explained_variance[:4] * 100):
    ax3.text(np.arange(1, len(explained_variance[:4]) + 1)[idx], (explained_variance[:4] * 100)[idx],
                  '{:.4f}'.format(txt))
ax3.grid()
ax3.set_title('Contribution of Loadings')
ax3.set_xlabel('PC#')

plt.show()

if __name__ == '__main__':
    plt.show()
