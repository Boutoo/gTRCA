# %% Imports
import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy

# %% Functions
def print_progress_bar(iteration, total, fill = '•', length=40, autosize = False):
    """
    Call in a loop to create terminal progress bar (Original Fill █)

    Args:
        iteration  (Int): current iteration (Required)
        total (int): total iterations (Required)
        fill (str): bar fill character (Optional)
        length (100): character length of bar (Optional)
        autosize (bool): automatically resize the length of the progress bar
        to the terminal window (Optional)

    Examples:
        >>> print_progres_bar(0,10)
        >>> for i in range(10):
        >>>     print_progress_bar(i,10)
        ∙ |••••••••••∙∙∙∙∙∙∙∙| 50% ∙

    """
    percent = ("{0:." + "0" + "f}").format(100 * (iteration / float(total)))
    styling = '%s |%s| %s%% %s' % ('∙', fill, percent, '∙')
    if autosize:
        cols, _ = shutil.get_terminal_size(fallback = (length, 1))
        length = cols - len(styling)
    filled_length = int(length * iteration // total)
    progress_bar = fill * filled_length + '-' * (length - filled_length)
    print('\r%s' % styling.replace(fill, progress_bar), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()

# %% Defining Class
class gTRCA() :
    def __init__(self, data, ncomps=5, time_window=None,
                 protocol_info=None, reg=10**5, normalize_q=True,
                 show_progress=True, verbose=True):
        self.verbose=verbose
        self.show_progress = show_progress
        self.nsubjects = len(data)
        self.mne_infos = [sub.info for sub in data]
        self.nchannels = np.array([len(sub.info['ch_names']) for sub in data])
        self.times = data[0].times # If Group: assume all subjects have the same times
        for sub in data:
            if np.any(sub.times != self.times):
                raise Exception('Please use Epochs with same Time Window and sampling frequency.')
        self.tmin, self.tmax = self.times[0], self.times[-1]
        self.tau = len(self.times)
        epochs = self.group_to_array(data, normalize_q)
        self.u, self.v, self.q0, self.q = self.calculate_u_q(epochs)
        self.s = self.calculate_s(epochs)
        inv_q, self.Svd = self.apply_inverse_reg(reg)
        self.rho, self.vecs = self.apply_eigen_decomposition(inv_q,reg)
        if ncomps == 'all':
            ncomps = len(self.rho)
        self.ncomps = ncomps
        self.ydata, self.maps, self.w = self.project_results(epochs, components=self.ncomps)
        self.fix_ydata_orientation()

    def group_to_array(self, data, normalize_q=True):
        """group_to_array
        This function is used to convert a list of Subjects into a numpy array.
        Args:
            data (list of Subjects): List of MNE Epochs to apply gTRCA

        Returns:
            epochs (list of np.ndarray): List of Subjects Data in np.ndarray format
        """
        if normalize_q:
            if self.verbose:
                print('Normalizing data...')
            epochs=[]
            for i, sub in enumerate(data):
                if self.show_progress is True:
                    print_progress_bar(i, len(data)-1)
                sub = sub.get_data()
                ntrials, nchs, tau = np.shape(sub) # (ntrials, nchs, tau)
                for ch in range(nchs):
                    sub[:,ch,:] = (sub[:,ch,:]-np.mean(sub[:,ch,:]))/np.std(sub[:,ch,:])
                epochs.append(sub.data)
        else:
            epochs = [sub.get_data() for sub in data] # Get Epoches Data
        return epochs

    def calculate_u_q(self, epochs):
        # Epochs (ntrials, nchs, tau)
        u = []
        v = []
        q0 = []
        q = np.zeros([np.sum(self.nchannels), np.sum(self.nchannels)])
        if self.verbose:
            print('Calculating U Matrix...')
        for i, ep in enumerate(epochs):
            ep = np.array(ep) # Retrieving from memory
            if self.show_progress is True:
                print_progress_bar(i, self.nsubjects-1)
            raw = np.transpose(ep, (1,0,2)) # raw = channel x time x trial
            raw = raw.reshape(np.shape(raw)[0], -1) # concatenating all trials, raw = channel x time
            q0 += [(raw @ raw.T)/np.shape(raw)[1]]
            u += [np.mean(ep, axis=0)]
            ntrials = np.shape(ep)[0] # Searching for local Ntrials size
            vsum = 0
            for k in range(ntrials):
                vsum += (ep[k,:,:] @ ep[k,:,:].T)
            v += [vsum/ntrials]
        if self.verbose:
            print('Calculating Q Matrix...')
        for n in range(self.nsubjects):
            if self.show_progress is True:
                print_progress_bar(n, self.nsubjects-1)
            q[n*self.nchannels[n]:(n+1)*self.nchannels[n],
              n*self.nchannels[n]:(n+1)*self.nchannels[n]] = q0[n]
        return u, v, q0, q

    def calculate_s(self, epochs):
        # computation of S and Q matrices:
        s = np.zeros([np.sum(self.nchannels), np.sum(self.nchannels)])
        if self.verbose:
            print('Calculating S Matrix...')
        count = 0
        for a in range(self.nsubjects):
            for b in range(self.nsubjects):
                if self.show_progress is True:
                    print_progress_bar(count, (self.nsubjects**2)-1)
                count += 1
                if a==b: # Diagonal
                    ntrials = np.shape(epochs[a])[0]
                    stmp = 2*(ntrials/((ntrials-1)*self.tau))* \
                        ((self.u[a] @ self.u[a].T) - (self.v[a]/ntrials))
                else: # Off-Diagonal
                    stmp = 1/self.tau*(self.u[a] @ self.u[b].T)
                s[a*self.nchannels[a]:(a+1)*self.nchannels[a],
                  b*self.nchannels[b]:(b+1)*self.nchannels[b]] = stmp
        return s

    def regularize_matrix(self,S,reg):
        eps=np.finfo(float).eps
        ix1=np.where(np.abs(S)<1000*np.finfo(float).eps)[0] #removing null components
        ix2 = np.where((S[0:-1]/(eps+S[1:]))>reg)[0] #cut-off based on eingenvalue ratios
        ix=np.union1d(ix1,ix2)
        if len(ix)==0:
            reg_num=len(S)
        else:
            reg_num=np.min(ix)
        return reg_num
    
    def apply_inverse_reg(self, reg):
        if self.verbose:
            print('Regularizing via Ratio Eig > ('+str(reg)+')...')
        U, S, V = np.linalg.svd(self.q)
        reg_num=self.regularize_matrix(S,reg)
        inv_q = (V[0:reg_num, :].T* (1./S[0:reg_num])) @ U[:, 0:reg_num].T
        return inv_q, S

    def apply_eigen_decomposition(self, inv_q,reg):
        M = inv_q @ self.s
        rho, vecs = np.linalg.eig(M) 
        indx=np.argsort(np.real(rho));indx=indx[-1::-1]
        rho=rho[indx]
        vecs=vecs[:,indx]
        reg_num=self.regularize_matrix(np.abs(rho),reg)
        rho=rho[0:reg_num]
        vecs=vecs[:,0:reg_num]
        if np.max(abs(np.imag(rho)))!=0:
            raise NameError("Rho has complex values: check")
        rho=np.real(rho)
        vecs=np.real(vecs)
        return rho, vecs

    def project_results(self, epochs=None, components=5):       
        if self.verbose:
            print('Projecting Results...')
        nsubs = self.nsubjects
        tau = self.tau
        w = [[]]*nsubs
        ydata = [[]]*nsubs
        maps = [[]]*nsubs
        y1 = [[]]*nsubs
        if components>0:
            for i, ep in enumerate(epochs):
                ep = np.array(ep)
                nchs = self.nchannels[i]
                ntrials, _, _ = np.shape(ep)
                # Vectors for each subject
                w[i] = np.zeros([components, nchs])
                for c in range(components):
                    w[i][c,:] = self.vecs[i*nchs:(i+1)*nchs, c]
                
                # Building y_data
                ydata[i] = np.zeros([components, ntrials, tau])
                for c in range(components):
                    for k in range(ntrials):
                        ydata[i][c,k,:] = w[i][c,:].T @ ep[k,:,:]
                
                # Normalization
                for c in range(components):
                    cmean = np.mean(np.mean(ydata[i][c,:,:],axis=0)) #
                    cstd = np.std(np.mean(ydata[i][c,:,:], axis=0)) #
                    ydata[i][c,:,:] = (ydata[i][c,:,:]-cmean)/cstd #
                    # for k in range(ntrials):
                    #     ydata[i][c,k,:] = (ydata[i][c,k,:]-np.mean(ydata[i][c,k,:]))/np.std(ydata[i][c,k,:])

                # Creting Scalp Maps
                y1[i] = np.zeros([components, tau])
                maps[i] = np.zeros([components, nchs])
                for c in range(components):
                    maps[i][c,:] = self.q0[i] @ w[i][c,:]
                
           # maps = np.real(maps) Unnecessary
            if self.verbose:
                print('Done!')
        else:
            print('No significant components')
        return ydata, maps, w

    def plot_maps(self, component=0, subject=None, axes=None):
        if subject is None:
            mean = np.mean([self.maps[sub][component] for sub in range(self.nsubjects)], axis=0) # maps: [sub][comp](62) --> (62)
            mne.viz.plot_topomap(mean, self.mne_infos[0], axes=axes,show=False,cmap='jet');
        else:
            mne.viz.plot_topomap(self.maps[subject][component,:], self.mne_infos[subject], axes=axes,show=False,cmap='jet');
        return

    def plot_ydata(self, component=0, subject=None, axes=None, colors=['gray','red'],basecorr=None):
        if axes is None:
            fig, ax = plt.subplots()
        else:
            ax = axes
        if basecorr!=None:
            baseline=[s for s,st in enumerate(self.times) if (st>=basecorr[0] and st<=basecorr[1])]
        if subject is None:
            mean = np.zeros(self.tau)
            for i, sub in enumerate(self.ydata):
                s_mean = np.mean(sub[component,:,:], axis=0)
                if basecorr!=None:
                    s_mean=s_mean-np.mean(s_mean[baseline])
                ax.plot(self.times, s_mean, color=colors[0], linewidth=.75)
                mean += s_mean
            mean = mean/len(self.ydata)
            ax.plot(self.times, mean, color=colors[1], linewidth=1.25)
        else:
            mean = np.zeros(self.tau)
            for k in range(np.shape(self.ydata[subject])[1]):
                s=self.ydata[subject][component,k,:]
                if basecorr!=None:
                    s=s-np.mean(s[baseline])
                ax.plot(self.times, s, color=colors[0], linewidth=.25)
                mean += s
            mean=mean/np.shape(self.ydata[subject])[1]
            ax.plot(self.times, mean, color=colors[1], linewidth=1)
        if axes is None:
            return fig
        else:
            return

    def fix_ydata_orientation(self):
            ncomps = np.shape(self.ydata[0])[0]
            ynorm = [np.zeros([ncomps, self.tau])]*self.nsubjects
            for c in range(ncomps):
                ynorm = [np.mean(y[c,:,:], axis=0) for y in self.ydata]
                ynorm = [(y-np.mean(y))/np.std(y) for y in ynorm]
                mean_abs = [np.abs(y) for y in ynorm]
                mean_abs = np.mean(np.array(mean_abs),axis=0)
                peak, _ = scipy.signal.find_peaks(mean_abs, distance=len(mean_abs))
                for i in range(len(self.ydata)):
                    peak_sig = np.sign(ynorm[i][peak])
                    self.ydata[i][c,:,:] = peak_sig*self.ydata[i][c,:,:]
                    self.maps[i][c,:] = peak_sig*self.maps[i][c,:]
                    self.w[i][c,:] = peak_sig*self.w[i][c,:]
            return

    def make_evoked(self, comp=None, subject=None):
        if comp is None:
            comp = self.ncomps
        w = self.w # [subject](ncomps, nchannels)
        ydata = self.ydata # [subject](ncomps, ntrials, tau)
        if subject is None:
            subs_ymean = [np.mean(suby, axis=1) for suby in ydata]
            yevoked = np.zeros([self.nsubjects, self.nchannels, self.tau])
            for i in range(self.nsubjects):
                # Multiplying each subject spatial map and its respective component
                yevoked[i] = w[i].T @ subs_ymean[i]
            yevoked = np.mean(yevoked, axis=0) # Mean of Subjects
        else:
            sub_ymean = np.mean(ydata[subject], axis=1)
            yevoked = np.zeros([self.nchannels, self.tau])
            yevoked = w[subject].T @ sub_ymean
        yevoked = mne.EvokedArray(yevoked, self.mne_info, tmin=self.tmin) 
        return yevoked

    def project(self, subject, trial='all'):
        """Get predictive filter of new available data

        Args:
            subject (_type_): subject object. Please use same trial duration
                since we're using same tw_idx for selecting projection window
            trial ('all', int or list optional): which trial to use.
                Defaults to 'all'. If list, gets form trial[0]:trial[1]

        Returns:
            w (ncomps, nchannels): predictive spatial filter W(a+1)
            corrs (ncomps): correlation of newsub_ydata and mean_group_ydata
            maps (nchannels): 
            q: covmatrix
        """
        # Subject input
        sub = subject.get_data()
        times = subject.times
        ntrials, nchs, _ = np.shape(sub)

        # Trial input
        trial = 'all'
        if isinstance(trial,str):
            if trial == 'all':
                trial = [i for i in range(ntrials)]
        elif type(trial) == int:
            trial = [trial]

        # Getting only selected trials
        data = sub[trial, :, :]

        # Subject Operations
        raw = np.transpose(data, (1,0,2))
        raw = raw.reshape(np.shape(raw)[0],-1) # (nchannels, ntrials*tau)
        q = (raw @ raw.T)/np.shape(raw)[1] # (nchannels, nchannels)
        
        inv_q = np.linalg.inv(q) # (nchannels, nchannels)
        if np.max(np.imag(inv_q)) != 0:
            print('Q Inverse has imaginary values')

        # Group Operations
        if np.any(times != self.times):
            raise Exception('Please make sure that the new subject time window and sampling frequency match gTRCA subjects')
        tau = len(times)
        nsubs = self.nsubjects # nsubs
        u = self.u # (nsubjects, nchannels, tau)
        wg = self.w # [nsubs](ncomps, nchannels)
        yg = self.ydata # [nsubs](ncomps, ntrials, tau)
        yg = [np.mean(yg[sub], axis=1) for sub in range(nsubs)] # [nsubs](ncomps, tau)
        yg = np.mean([yg[sub] for sub in range(nsubs)], axis=0) # [nsubs](ncomps, tau)

        ncomps = self.ncomps
        
        # Applying Projection
        w = np.zeros([ncomps, nchs])
        sub_ydata = np.zeros([ncomps, tau])
        corrs_trials = np.zeros([ntrials,ncomps])
        corrs_avg=np.zeros(ncomps)
        maps = np.zeros([ncomps, nchs])
        for c in range(ncomps):
            M = np.sum([u[i].T @ wg[i][c,:] for i in range(nsubs)], axis=0)
            x = np.mean(data, axis=0) # (nchannels, tau)
            w[c,:] = (1/(2*nsubs))* ((inv_q @ x) @ M)
            sub_ydata[c,:] = w[c,:] @ x
            maps[c,:] = q @ w[c,:]  
            norm=np.std(sub_ydata[c,:])
            sub_ydata[c,:]=sub_ydata[c,:]/norm
            w[c,:] = w[c,:]/norm
            maps[c,:]=maps[c,:]/norm
            correl,prel=scipy.stats.pearsonr(sub_ydata[c,:], yg[c,:])
            if prel<0.05:
                corrs_avg[c] = correl

        return sub_ydata, corrs_trials, corrs_avg, maps, q, w