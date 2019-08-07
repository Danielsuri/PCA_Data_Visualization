import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import glob
from scipy.signal import savgol_filter
from scipy import signal

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

Egg = []
MCT5 = []
Oil = []
Protein = []
sam3 = []
sam1 = []
sam2 = []
Tap = []
yog = []

sample_names = abs_df.columns.values
for sample in sample_names:
    if "sample1" in sample:
        sam1.append(sample)
    if "sample2" in sample:
        sam2.append(sample)
    if "sample3" in sample:
        sam3.append(sample)
    if "Egg" in sample:
        Egg.append(sample)
    if "MCT10" in sample:
        MCT5.append(sample)
    if "Oil" in sample:
        Oil.append(sample)
    if "Protein" in sample:
        Protein.append(sample)
    if "Tap" in sample:
        Tap.append(sample)
    if "Yog" in sample:
        yog.append(sample)

mean_sam1 = abs_df[sam1].mean(1)
mean_sam2 = abs_df[sam2].mean(1)
mean_sam3 = abs_df[sam3].mean(1)
mean_Egg = abs_df[Egg].mean(1)
mean_Tap = abs_df[Tap].mean(1)
mean_yog = abs_df[yog].mean(1)
mean_oil = abs_df[Oil].mean(1)
mean_MCT5 = abs_df[MCT5].mean(1)
mean_protein = abs_df[Protein].mean(1)

mean_df = pd.DataFrame()
mean_df.insert(loc=0, column="Protein Powder", value=mean_protein)
mean_df.insert(loc=0, column="Sunflower oil", value=mean_oil)
mean_df.insert(loc=0, column="Tap Water", value=mean_Tap)
mean_df.insert(loc=0, column="Egg", value=mean_Egg)
mean_df.insert(loc=0, column="sample3", value=mean_sam3)
mean_df.insert(loc=0, column="Protein Yogurt", value=mean_yog)
mean_df.insert(loc=0, column="sample2", value=mean_sam2)
mean_df.insert(loc=0, column="MCT 10", value=mean_MCT5)
mean_df.insert(loc=0, column="sample1", value=mean_sam1)

plot_df = pd.DataFrame()
plot_df.insert(loc=0, column="Protein Powder", value=mean_protein)
plot_df.insert(loc=0, column="Egg", value=mean_Egg)
plot_df.insert(loc=0, column="sample3", value=mean_sam3)
plot_df.insert(loc=0, column="sample2", value=mean_sam2)
plot_df.insert(loc=0, column="MCT 10", value=mean_MCT5)
plot_df.insert(loc=0, column="sample1", value=mean_sam1)
plot_sg_df = pd.DataFrame(data=savgol_filter(plot_df.values, 5, 3), columns=plot_df.columns,
                     index=plot_df.index)
plot_sg_df_diff = plot_sg_df.diff().plot(title='Sav_Gol First derivative')


samples_to_plot = []
# Choose sample to plot: 'NQ' for all samples
str_to_find = ''

sample_names = mean_df.columns.values
for sample in sample_names:
    if str_to_find in sample:
        samples_to_plot.append(sample)

# sav_gol dataframe
sg_df = pd.DataFrame(data=savgol_filter(mean_df.values, 5, 3), columns=mean_df.columns,
                     index=mean_df.index)

# Plots
if len(samples_to_plot) == 1:
    # Find peaks in signal
    peaks_idxs = signal.find_peaks(sg_df[samples_to_plot[0]].values, distance=80)
    peak_wavelen = sg_df[samples_to_plot[0]].index.values[[peaks_idxs[0]]]
    peaks_vals = sg_df[samples_to_plot[0]].values[peaks_idxs[0]]
    peaks_diff = np.abs(np.diff(peaks_vals))
    print(peaks_diff)
    # create plot with scatter on peaks
    ax = sg_df[samples_to_plot].plot(title=sg_df[samples_to_plot].columns.values[0])
    ax.scatter(peak_wavelen, peaks_vals, marker='o', c='red')
    for idx, items in enumerate(peaks_diff):
        ax.vlines(x=(peak_wavelen[idx] + peak_wavelen[idx + 1]) / 2, ymin=min(peaks_vals[idx], peaks_vals[idx + 1]),
                   ymax=max(peaks_vals[idx], peaks_vals[idx + 1]), linestyles='dashed', label=peaks_diff[idx])
        ax.text(x=(peak_wavelen[idx] + peak_wavelen[idx + 1]) / 2, y=min(peaks_vals[idx], peaks_vals[idx + 1]),
                s='%.2f' % peaks_diff[idx])
    ax.set_ylabel('Absorbance')

    # sg_df[samples_to_plot].plot(title=str_to_find + ' Sav_Gol Absorbance plot')
    sg_df[samples_to_plot].diff().plot(title=sg_df[samples_to_plot].columns.values[0] + ' Sav_Gol First derivative')

    # absorbance_df[samples_to_plot].plot(title=str_to_find + ' Absorbance plot')
    # absorbance_df[samples_to_plot].diff().plot(title=str_to_find + ' First derivative')
else:
    # Sav_Gol plots
    sg_df[samples_to_plot].plot(title=str_to_find + ' Sav_Gol Absorbance plot')
    sg_df[samples_to_plot].diff().plot(title=str_to_find + ' Sav_Gol First derivative')

    mean_df[samples_to_plot].plot(title=str_to_find + ' Absorbance plot')
    mean_df[samples_to_plot].diff().plot(title=str_to_find + ' First derivative')

plt.show()
