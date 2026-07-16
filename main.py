"""
Find the peak frequency within a frequency band from a PSD table.

This app loads a per-channel PSD table (TSV), finds the dominant peak
frequency per channel within a given band using MNE's peak-finder, and
saves the result plus diagnostic plots.

Inputs:
    - psd: Path to per-channel PSD TSV file
    - fmin, fmax: Frequency band bounds

Outputs:
    - out_dir/psd.tsv: Per-channel peak frequency table
    - out_figs/psd_allchannels.png: Per-channel PSD with detected peaks
    - out_figs/psd_peak_frequency.png: PSD with mean peak frequency marked
    - out_figs/hist_peak_frequency.png: Histogram of peak frequencies
    - product.json: Metadata about the computed peaks
"""

# Copyright (c) 2026 brainlife.io
#
# Author: Guiomar Niso

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'brainlife_utils'))

# Standard imports
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns

# Import shared utilities
from brainlife_utils import (
    load_config,
    setup_matplotlib_backend,
    ensure_output_dirs,
    create_product_json,
    add_info_to_product,
    add_image_to_product,
    require_config_keys
)

# Set up matplotlib for headless execution
setup_matplotlib_backend()

# Ensure output directories exist
ensure_output_dirs('out_dir', 'out_figs')

# Load configuration
config = load_config()
require_config_keys(config, ['psd', 'fmin', 'fmax'])

# == GET CONFIG VALUES ==
fname = config['psd']
fmin = config['fmin']
fmax = config['fmax']

# == LOAD DATA ==
df_psd = pd.read_csv(fname, sep='\t')
canales = df_psd['channels'].copy()

# Number of frequencies computed for the PSD
nfreqs = df_psd.shape[1]
df = df_psd.iloc[:, 1:nfreqs].copy()  # To avoid the case where changing df also changes df_psd
# List of frequencies
freqs = df.columns.to_numpy()
freqs = freqs.astype(float)
# PSD values
psd_welch = df.to_numpy()
# Number of channels
nchannels = psd_welch.shape[0]

# Extract the frequencies that fall inside the band
ifreqs = [i for i, f in zip(range(0, len(freqs)), freqs) if f > fmin and f < fmax]
band_freqs = np.take(freqs, ifreqs)

# ==== FIND FREQUENCY PEAK ====

channel_peak = []

# Prepare for Figure 1 containing all the channels
plt.figure(1)

if nchannels == 1:
    axs = [0]
    fig, axs1 = plt.subplots(nchannels, 1, facecolor='w', edgecolor='k')
    axs[0] = axs1
    dpi = 200
else:
    Nfigs = 10
    fig, axs = plt.subplots(math.ceil(nchannels/Nfigs), Nfigs, figsize=(40, math.ceil(nchannels/Nfigs*2)), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.5, wspace=.2)
    axs = axs.ravel()
    dpi = 20

for channel in range(0, nchannels):

    # Find maxima with noise-tolerant fast peak-finding algorithm
    psd_channel = np.take(psd_welch[channel, :], ifreqs)
    pic_loc, pic_mag = mne.preprocessing.peak_finder(psd_channel, extrema=1, verbose=None)

    # From all the peaks found, get the main peak

    # If one peak found
    if pic_loc.size == 1:
        peak = pic_loc[0].copy()
        if peak == 0: peak = math.nan  # if it's the first value, then ignore, maybe just a decreasing curve

    # If no peak found
    if pic_loc.size == 0:
        peak = math.nan
        print('No peak found for channel: ', canales[channel])

    # If more than one peak found
    elif pic_loc.size > 1:
        peak = np.where(psd_channel == max(pic_mag))[0][0]  # take the max
        if peak == 0: peak = pic_loc[np.argmax(pic_mag[1:,])+1]  # if it's the first value, take the next max
        if peak == psd_channel.size-1: peak = pic_loc[np.argmax(pic_mag[0:-1])]  # if it's the last value, take the next max

        print('Multiple peaks found for channel: ', canales[channel])

    # Get the frequency of the peak
    pic_freq = np.take(band_freqs, peak) if not math.isnan(peak) else math.nan
    channel_peak.append(pic_freq)

    # FIGURE 1
    axs[channel].plot(band_freqs, psd_channel)
    axs[channel].plot(np.take(band_freqs, pic_loc), pic_mag, '*')
    axs[channel].axvline(x=pic_freq, c='k', ls=':')
    axs[channel].set_title(canales[channel])
    axs[channel].set_xlim(fmin, fmax)

# Delete empty axes
if nchannels > 1:
    for i in range(nchannels, math.ceil(nchannels/Nfigs)*Nfigs):
        fig.delaxes(axs.flatten()[i])

# Save Figure 1
allchannels_path = os.path.join('out_figs', 'psd_allchannels.png')
plt.savefig(allchannels_path, dpi=dpi)
plt.close()

# Average value across all channels
mean_peak = np.nanmean(channel_peak, axis=0)

# == SAVE FILE ==
df_alpha = pd.DataFrame(channel_peak, index=canales, columns=['peak'])
tsv_path = os.path.join('out_dir', 'psd.tsv')
df_alpha.to_csv(tsv_path, sep='\t')

# ==== PLOT FIGURES ====

# FIGURE 2
plt.figure(2)
plt.plot(freqs, psd_welch.transpose(), zorder=1)
plt.xlim(xmin=0, xmax=max(freqs))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density')

plt.axvline(x=mean_peak, c='k', ls=':')
peak_freq_path = os.path.join('out_figs', 'psd_peak_frequency.png')
plt.savefig(peak_freq_path)
plt.close()

# FIGURE 3
plt.figure(3)
sns.set_theme(style="ticks")

if nchannels == 1:
    sns.histplot(data=channel_peak, kde=True)  # doesn't allow binwidth=0.25 for one value data
else:
    sns.histplot(data=channel_peak, binwidth=0.25, kde=True, kde_kws={'cut': 10})

plt.xlim(xmin=fmin, xmax=fmax)
plt.xlabel('Peak frequency (Hz)')
sns.despine()
hist_path = os.path.join('out_figs', 'hist_peak_frequency.png')
plt.savefig(hist_path)
plt.close()

# == CREATE PRODUCT.JSON ==
product_items = []
add_info_to_product(product_items, f'Mean peak frequency across channels ({fmin}-{fmax} Hz): {mean_peak:.4g} Hz', 'success')
add_image_to_product(product_items, 'PSD per channel', filepath=allchannels_path)
add_image_to_product(product_items, 'Peak frequency histogram', filepath=hist_path)
create_product_json(product_items)
