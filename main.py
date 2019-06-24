import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import glob
from numpy.random import rand
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.image import AxesImage

# loading dataset into Pandas DataFrame

all_files = glob.glob('PCA/*.txt')
all_files.sort()
df2 = pd.DataFrame()

for filename in all_files:
    col_name = filename.split("\\")[1].split('.')[0]
    data = pd.read_csv(filename, sep='\t', skiprows=13, names=['Wave_Length', col_name])
    try:  # insert Wave Length only once:
        df2.insert(loc=0, column='Wave_Length', value=data['Wave_Length'].values, allow_duplicates=False)
        pass
    except ValueError:
        pass
    df2.insert(loc=1, column=col_name, value=data[col_name].values)

df2 = df2.T
for names in df2.index.values:
    df2.rename(index={names: names.split('_')[0]}, inplace=True)

x = StandardScaler().fit_transform(df2.iloc[1:].values)
y = df2.iloc[1:].index.values

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data=principalComponents
                           , columns=['PC1', 'PC2'])

principalDf.insert(loc=0, column='targets', value=y)

explained_variance = pca.explained_variance_ratio_
loadingsDf = pd.DataFrame(data=pca.components_.T * np.sqrt(pca.explained_variance_),
                          columns=['PC1 Loadings', 'PC2 Loadings'])


# plot:
def line_picker(line, mouseevent):
    """
    find the points within a certain distance from the mouseclick in
    data coords and attach some extra attributes, pickx and picky
    which are the data points that were picked
    """
    if mouseevent.xdata is None:
        return False, dict()
    xdata = line.get_xdata()
    ydata = line.get_ydata()
    lbl = line.get_label()
    maxd = 0.25
    d = np.sqrt(
        (xdata - mouseevent.xdata) ** 2 + (ydata - mouseevent.ydata) ** 2)

    ind, = np.nonzero(d <= maxd)
    if len(ind):
        pickx = xdata[ind]
        picky = ydata[ind]
        picklbl = lbl
        props = dict(ind=ind, pickx=pickx, picky=picky, picklbl=picklbl)
        return True, props
    else:
        return False, dict()


def onpick2(event):
    print('onpick2 line:', event.pickx, event.picky, event.picklbl)


targets = list(set(list(principalDf['targets'])))
targets.sort()
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
font_size = 10
for items in targets:
    x = principalDf.loc[principalDf['targets'] == items, 'PC1'].values
    y = principalDf.loc[principalDf['targets'] == items, 'PC2'].values
    line, = ax[0].plot(x, y, 'o', picker=line_picker, label=items)

fig.canvas.mpl_connect('pick_event', onpick2)
ax[0].legend()
ax[0].set_xlabel('PC1', fontsize=font_size)
ax[0].set_ylabel('PC2', fontsize=font_size)
ax[0].set_title('2 Component PCA')
ax[0].grid()

ax[1].plot(df2.iloc[0].values, loadingsDf['PC1 Loadings'].values, label=loadingsDf.columns.values[0])
ax[1].plot(df2.iloc[0].values, loadingsDf['PC2 Loadings'].values, label=loadingsDf.columns.values[1])
ax[1].set_xlabel('Wave Length', fontsize=font_size)
ax[1].set_ylabel('Loadings', fontsize=font_size)
ax[1].set_title('Loadings')
ax[1].legend()
ax[1].grid()
plt.subplots_adjust(wspace=0.3)

if __name__ == '__main__':
    plt.show()
