# %% Imports
import mne
import matplotlib.pyplot as plt
import numpy as np
from gtrca import gTRCA, create_surrogate
from scipy.io import loadmat

PATH = "C:/Users/bruno/Documents/NNCLab/Data/SHAM/"
FILES = ["AM_Session_4_PremotorL_TMS_EVOKED.mat",
         "AP_Session_2_PremotorL_TMS_EVOKED.mat"]

file = PATH+FILES[0]

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
for file in FILES:
    epochs+=[mat_to_epochs(PATH+file, [-.2,.5])]

gtrca = gTRCA(epochs, ncomps=1)

# %% Analysis
ydata = [np.mean(sub[0,:,:], axis=0) for sub in gtrca.ydata]

new_sub_pmtms = mat_to_epochs(PATH+'AV_Session_4_PremotorL_TMS_EVOKED.mat', [-.2,.5])
new_sub_sham = mat_to_epochs(PATH+'AV_Session_9_MotorL_ShamNoiseON_EVOKED.mat', [-.2,.5])

results = gtrca.project(new_sub_pmtms)
results2 = gtrca.project(new_sub_sham)

plt.plot(np.array(ydata).T,'gray');
plt.plot(results[0][0,:],'red');
plt.plot(results2[0][0,:],'blue', alpha=.5);

gtrca.plot_maps(0,0)
gtrca.plot_maps(0,1)
mne.viz.plot_topomap(results[3][0], new_sub_pmtms.info);
mne.viz.plot_topomap(results2[3][0], new_sub_sham.info);

# %% Testing Surrogate
times = epochs[0].times
nsubs = len(epochs)
sub_range = np.arange(nsubs)
trial_surr = create_surrogate(epochs, mode='trial')
subject_surr = create_surrogate(epochs, mode='subject')

def get_imagesc(epochs):
    nsubs = len(epochs)
    nchs = [len(sub.info['ch_names']) for sub in epochs]
    ch_cumsum = np.concatenate([np.zeros(1),np.cumsum(nchs)])
    times = epochs[0].times
    image = np.zeros([np.sum(nchs), len(times)])
    for i,sub in enumerate(epochs):
        image[int(ch_cumsum[i]):int(ch_cumsum[i+1]),:] = sub.average().get_data()
    return image

fig, axs = plt.subplots(3,1,dpi=250,sharex=True,layout='constrained');
axs[0].set_title('Epochs')
axs[0].imshow(get_imagesc(epochs), cmap='jet', aspect='auto', interpolation='none', extent=(times[0], times[-1], nsubs-0.5, -0.5));
axs[1].set_title('Trial-based Shifting')
axs[1].imshow(get_imagesc(trial_surr), cmap='jet', aspect='auto', interpolation='none', extent=(times[0], times[-1], nsubs-0.5, -0.5));
axs[2].set_title('Subject-based Shifting')
axs[2].imshow(get_imagesc(subject_surr), cmap='jet', aspect='auto', interpolation='none', extent=(times[0], times[-1], nsubs-0.5, -0.5));

for ax in axs:
    ax.set_yticks(sub_range)