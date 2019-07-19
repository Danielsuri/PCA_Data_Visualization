import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import glob
from scipy.signal import savgol_filter
from scipy import signal

all_files = glob.glob('All_samples/*.txt')
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

air_sam = []
str_to_find = 'Air'

sample_names = df.columns.values
for sample in sample_names:
    if str_to_find in sample:
        air_sam.append(sample)
# Mean over all air samples
mean_air_sam = df[air_sam].mean(1)
# Remove all air samples to stay only with measurements data
df_minus_dark.drop(columns=air_sam, inplace=True)
# Divide by air mean => Absorbance dataframe
absorbance_df = df_minus_dark.div(mean_air_sam, axis=0)

samples_to_plot = []
# Choose sample to plot: 'NQ' for all samples
str_to_find = 'Yog'

sample_names = absorbance_df.columns.values
for sample in sample_names:
    if str_to_find in sample:
        samples_to_plot.append(sample)


# sav_gol dataframe
sg_df = pd.DataFrame(data=savgol_filter(absorbance_df.values, 5, 3), columns=absorbance_df.columns,
                     index=absorbance_df.index)

# Sav_Gol plots
sg_df[samples_to_plot].plot(title=str_to_find + ' Sav_Gol Absorbance plot')
sg_df[samples_to_plot].diff().plot(title=str_to_find + ' Sav_Gol First derivative')

absorbance_df[samples_to_plot].plot(title=str_to_find + ' Absorbance plot')
absorbance_df[samples_to_plot].diff().plot(title=str_to_find + ' First derivative')

plt.show()
