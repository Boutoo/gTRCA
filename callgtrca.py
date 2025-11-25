"""
Example script to apply gTRCA to the TMS–EEG datasets from Couto et al. (2025).

- Loads preprocessed MNE epochs for all subjects (Milan and Aalborg datasets)
- Fits the gTRCA model at the group level
- Plots:
    * Average component waveforms across subjects
    * Individual subject spatial maps (topographies) for each component

The script assumes that:
- The gTRCA library is installed and importable as `from gtrca import gTRCA`
- The data are stored as MNE Epochs objects in .fif format, one file per subject (see Couto et al., 2025)

"""


import mne
from gtrca import gTRCA
import numpy as np
import matplotlib.pyplot as plt

def load_epochs(PATHTOEPOCHS,nsubjects,selsubjects='All',trials=None,time_window=None):
    """
    Load preprocessed MNE epochs for multiple subjects.

    Parameters
    ----------
    PATHTOEPOCHS : str
        Path to the folder containing the epochs in .fif format.
    nsubjects : int
        Total number of subjects available in the dataset.
    selsubjects : list or str, optional
        List of subject identifiers to load, or 'All' to load all subjects (default).
    trials : int, optional
        If specified, only the first 'trials' epochs will be selected.
    time_window : list or tuple, optional
        If specified, crop each epoch to this [tmin, tmax] time window (in seconds).

    Returns
    -------
    epochs : list of mne.Epochs
        A list containing the loaded (and optionally cropped) epochs for each subject.
    """
    
    # If selsubjects is 'All', build the list of all subject IDs: S01, S02, ...
    if selsubjects=='All':
        subjects = [f'S{i:02}' for i in range(1, nsubjects + 1)]
    else:
        subjects=selsubjects

    epochs=[]
    for s in subjects:
        e = mne.read_epochs(f'{PATHTOEPOCHS}/{s}_epo.fif')
        e.info['subject_info']={'Name':s}
        if trials is not None:
            e=e[0:trials]
            print(f'Selecting first {trials} trials for subject {s}')
        if time_window is not None:
            e.crop(tmin=time_window[0],tmax=time_window[1])
            print(f'Cropping subject {s} to window {time_window}')
        epochs.append(e)

    return epochs

def plot_avgcomponent(gtrca, component=0,title=None):
    """
    Plot the average temporal waveform and topography of a given GTRCA component.

    Parameters
    ----------
    gtrca : gTRCA
        Fitted gTRCA object.
    component : int, optional
        Index of the component to visualize (default is 0).
    title : str, optional
        Title to include in the plot.
    """
    # Get normalized and oriented components for all subjects
    # y: list/array of time courses; maps: spatial maps (per subject)
    y, maps = gtrca.get_component(component=[component], average=True,orientation_window=[0,0.1])

    # Average spatial map across subjects
    maps_avg=np.mean(maps,axis=0)

    fig, ax = plt.subplots(2,1,figsize=(6, 10))
    # Plot individual subject time courses (faint lines)
    for i in y:
        ax[0].plot(gtrca.times, i[0,:], 'black', alpha=.2)
    # Plot the average across subjects (thicker line)
    ax[0].plot(gtrca.times, np.mean(y,axis=0)[0,:], 'black', linewidth=2)
    ax[0].set_xlim([-0.1,0.5])
    ax[0].set_xlabel('Time (ms)')
    ax[0].set_ylabel('n.u.')
    ax[0].set_title(f'{title} - Avg Component {component}')
    
    # Plot average topography of the component
    mne.viz.plot_topomap(maps_avg[0,:], gtrca.infos[0], cmap='RdBu_r', axes=ax[1], show=False)
    ax[1].set_title(f'lambda = {gtrca.eigenvalues[component]:.3f}; lambda_A={gtrca.eigenvalues[component]/len(y):.3f}')

    plt.tight_layout()
    plt.show()

def plot_individualspatialmaps(gtrca,component=0,title=None):

    """
    Plot spatial maps (topographies) of a GTRCA component for all individual subjects.

    Parameters
    ----------
    gtrca : gTRCA
        Fitted gTRCA object.
    component : int, optional
        Index of the component to visualize (default is 0).
    title : str, optional
        Title to include in the figure (unused in plot).
    """

    n=len(gtrca.infos)

    # Retrieve component spatial maps for all subjects    
    _,Y=gtrca.get_component(average=True,component=[component],orientation_window=[0,0.1])

    # Arrange individual topographies in a 2-row grid
    cols = int(np.ceil(n / 2))
    fig, _ = plt.subplots(figsize=(cols, 2), dpi=500, constrained_layout=True)
    gs = fig.add_gridspec(2, cols)
    plt.axis('off')

    ax_list = []
    for i in range(n):
        row, col = divmod(i, cols)
        ax = fig.add_subplot(gs[row, col])
        ax.set_title(f'S{i}', fontsize=8)
        mne.viz.plot_topomap(Y[i][0, :], gtrca.infos[i], axes=ax, show=False, cmap='RdBu_r')
        ax_list.append(ax)

    plt.show()


# =============================================================================
# Set main folder and run example on Milan and Aalborg datasets
# =============================================================================

# To use this script:
# 1) Download the two .zip files containing the Milan and Aalborg datasets (see Couto et al., 2025)
# 2) Extract both into the same parent directory ("mainfolder"), which should
#    contain the subfolders 'Dataset_Milan' and 'Dataset_Aalborg'.
# 3) Set the path to your local mainfolder below:
mainfolder='/Data/Coutoetal2025/' # contains 'Dataset_Milan' and 'Dataset_Aalborg'



# -------------------------
# Milan dataset
# -------------------------

Dataset='Dataset_Milan'
PATHTOEPOCHS=mainfolder+Dataset

# Load MNE Epochs for all 16 subjects
epochs=load_epochs(PATHTOEPOCHS,nsubjects=16)


# Fit gTRCA at the group level
gtrca=gTRCA()
gtrca.fit(epochs)

# Plot first three components: average waveform + individual spatial maps
for s in range(3):
    plot_avgcomponent(gtrca,component=s,title=Dataset)
    plot_individualspatialmaps(gtrca,component=s,title=Dataset)


# -------------------------
# Aalborg dataset
# -------------------------

Dataset='Dataset_Aalborg'
PATHTOEPOCHS=mainfolder+Dataset

# Load MNE Epochs for all 22 subjects
epochs=load_epochs(PATHTOEPOCHS,nsubjects=22)

# Fit gTRCA at the group level
gtrca=gTRCA()
gtrca.fit(epochs)

# Plot first three components: average waveform + individual spatial maps
for s in range(3):
    plot_avgcomponent(gtrca,component=s,title=Dataset)
    plot_individualspatialmaps(gtrca,component=s,title=Dataset)
