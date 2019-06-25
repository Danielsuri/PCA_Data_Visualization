import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

all_files = glob.glob('plot/*.txt')
all_files.sort()

df = pd.DataFrame()
for file_name in all_files:
    col_name = file_name.split("\\")[1].split('.')[0]
    data = pd.read_csv(file_name, sep='\t', skiprows=13, names=['Wave_Length', col_name])
    try:
        df.insert(loc=0, column='Wave_Length', value=data['Wave_Length'].values, allow_duplicates=False)
        pass
    except ValueError:
        pass
    df.insert(loc=1, column=col_name, value=data[col_name].values)

samples_to_plot = []
str_to_find = 'Egg'
sample_names = df.columns.values
for sample in sample_names:
    if str_to_find in sample:
        samples_to_plot.append(sample)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
for items in samples_to_plot:
    ax.plot(df['Wave_Length'].values, df[items].values, label=items)

ax.set_xlabel('Wave_Length')
ax.set_ylabel('Intensity')
ax.grid()
ax.legend()
plt.show()
