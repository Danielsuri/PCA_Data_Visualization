import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import glob

all_files = glob.glob('PCA/*.txt')
all_files.sort()

df = pd.DataFrame()
for file_name in all_files:
    col_name = file_name.split("\\")[1].split('.')[0]
    data = pd.read_csv(file_name, sep='\t', skiprows=13, names=['Wave_Length', col_name])
    df.insert(loc=0, column=col_name, value=data[col_name].values)

df = df.T
df.columns = data['Wave_Length']
df = df.T
samples_to_plot = []
# Choose sample to plot:
str_to_find = 'Sample'

sample_names = df.columns.values
for sample in sample_names:
    if str_to_find in sample:
        samples_to_plot.append(sample)

df[samples_to_plot].plot(title=str_to_find + ' sample plot')

df[samples_to_plot].diff().plot(title=str_to_find + ' First derivative')
plt.show()
