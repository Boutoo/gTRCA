# %% Imports
import sys
import os
sys.path.append('../')
import mne
from gtrca import gTRCA, create_surrogate
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# %% Loading Data
PATH = "C:/Users/bruno/Documents/NNCLab/Data/SHAM/"

# Search for files
files = os.listdir(PATH)
files = [f for f in files if f.endswith('MotorL_TMS_EVOKED.mat')]

# Create MNE Epochs from files

def mat_to_epochs(file, time_window=None, downsample=False, ch_labels=None, montage='easycap-M1'):
    """ This function reads EEG data with .mat format.
        Specially made to deal with Milan SHAM Protocol data.

    Args:
        file (str): Full file path
        time_window (list or None, optional): Time to segment epochs in seconds (e.g. [-.2, .5]). Defaults to None.
        downsample (bool, optional): Downsample data (True or False). Defaults to False.
        ch_labels (_type_, optional): Label of channels defaults to Milano SHAM Protocol format. Defaults to None.
        montage (str, optional): MNE's built-in montage to use. Defaults to 'easycap-M1'.
    """
    data_mat = loadmat(file)
    data_mat['times'] = data_mat['times'][0]
    data = np.array(data_mat['Y'])
    data = np.transpose(data, [2,0,1]) # Mne default is (ntrials, nchannels, nsamples)
    times = np.array(data_mat['times'])/1000 #in s        
    if downsample!=False:
        data=data[:,:,range(0,len(times),downsample)]
        times=times[range(0,len(times),downsample)]
    sfreq = 1/(times[1]-times[0])
    tmin = times[0]
    if 'origem' in data_mat:
        origem = data_mat['origem']
    else:
        origem='none'           
    # Creating MNE Instances:
    if ch_labels is None: # Sets to default
        ch_labels = ['Fp1','Fpz','Fp2','AF3','AFz','AF4','F7','F5',
                            'F3','F1','Fz','F2','F4','F6','F8','FT7','FC5',
                            'FC3','FC1','FCz','FC2','FC4','FC6','FT8','T7',
                            'C5','C3','C1','Cz','C2', 'C4','C6','T8','TP9',
                            'TP7','CP5','CP3','CP1','CPz','CP2','CP4','CP6',
                            'TP8','TP10','P7','P5','P3','P1','Pz','P2','P4',
                            'P6','P8','PO7','PO3','POz','PO4','PO8','O1',
                            'Oz','O2','Iz']
    else:
        ch_labels=ch_labels

    if 'channels' in data_mat:
        ix=[ch_labels[z-1] for z in data_mat['channels'][0]]
        ch_labels=ix

    # Creating Info
    mne_info = mne.create_info(ch_names=ch_labels, sfreq=sfreq,
                                ch_types='eeg')
    ntrials = np.shape(data)[0]
    if time_window is None:
        idx = [0,-1]
        tmin = tmin
    else:
        idx = [np.abs(np.asarray(times)-time_window[i]).argmin() for i in range(len(time_window))]
        tmin = times[idx[0]]
    epochs = mne.EpochsArray(data[:,:,idx[0]:idx[1]], mne_info, tmin=tmin, verbose=False);
    if montage != None:          
        epochs.set_montage(montage)
    return epochs

epochs = []
for f in files:
    epochs.append(mat_to_epochs(PATH+f, time_window=[-.1, .5]))

# %% Chosing first 3 components (since threshold is still the same)
gtrca = gTRCA(epochs, n_components=3)

# %% Visualizing components
fig, axs = plt.subplots(3, 1, dpi=250, layout='constrained', sharey=True)
for i, ax in enumerate(axs):
    ydata = [np.mean(sub[i,:,:], axis=0) for sub in gtrca.ydata]
    ax.plot(gtrca.times, np.array(ydata).T, 'gray')
    ax.plot(gtrca.times, np.mean(ydata, axis=0).T, 'r', lw=2)
fig.savefig(f'MotorL_TMS_EVOKED.eps')

# %% Topographies
for i in range(3):
    gtrca.plot_maps(i)
    plt.title(f'Component {i}')
    fig = plt.gcf()
    fig.savefig(f'topo_comp{i}.eps')

# %% Correlations

def plot_corrs_distributions(gtrca, comp, save=True):
    # np.triu_indices(tam, k=1)
    fig, axs = plt.subplots(1,2,dpi=250, constrained_layout=True)
    components = np.array([np.mean(sub_data[comp,:,:], axis=0) \
                        for sub_data in gtrca.ydata]) # Gets component i from all subs
    maps = [maps[comp,:] for maps in gtrca.maps]
    components = np.corrcoef(components)
    maps = np.corrcoef(maps)
    idxs = np.triu_indices(np.shape(components)[0], k=1)
    axs[0].set_title('Temporal Covariance')
    axs[0].hist(components[idxs], bins=20, label='components')
    axs[0].axvline(np.median(components[idxs]), color='red')
    axs[1].set_title('Spatial Covariance')
    axs[1].hist(maps[idxs], bins=20, label='maps')
    axs[1].axvline(np.median(maps[idxs]), color='red')
    fig.suptitle('Correlations between components')
    if save:
        fig.savefig(f'correlations_comp{comp}.eps')

for i in range(3):
    plot_corrs_distributions(gtrca, i)
