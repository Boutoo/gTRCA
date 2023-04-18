# %% Imports
import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy
import copy as cp

# %% Functions
def print_progress_bar(iteration, total, fill = '•', length=40):
    """
    Create a progress bar to display the progress of a loop.

    Args:
        iteration  (Int): current iteration (Required)
        total (int): total iterations (Required)
        fill (str): bar fill character (Optional)
        length (100): character length of bar (Optional)
        to the terminal window (Optional)
    """
    percent = ("{0:." + "0" + "f}").format(100 * (iteration / float(total)))
    styling = '%s |%s| %s%% %s' % ('∙', fill, percent, '∙')
    filled_length = int(length * iteration // total)
    progress_bar = fill * filled_length + '-' * (length - filled_length)
    print('\r%s' % styling.replace(fill, progress_bar), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()

# Create 1 Surrogates (chose what kind)
def create_surrogate(epochs, mode='trial', minjitter=0, maxjitter='nsamples'):
    """ Creates a surrogate of the data by shifting the data in time. Data must be a list of mne.Epochs(...) format.

    Args:
        epochs (list): List of mne.Epochs(...) to create a surrogate from.
        mode ('trial' or 'subject', optional): What mode to use: trial or subject-based shifting. Defaults to 'trial'.
        minjitter (int, optional): Minimum jitter in samples. Defaults to 0.
        maxjitter (int or 'nsamples', optional): Maximum jitter in samples. Defaults to 'nsamples'.

    Raises:
        Please use same time window and sampling frequency for all subjects:
            Make sure that all Epochs have the same time window and sampling frequency.
        Please use a int as maxjitter of default 'nsamples':
            Make sure that maxjitter is a int or 'nsamples'

    Returns:
        surrogate (list): List of surrogated mne.Epochs(...) data.
    """
    # Fetching Epochs
    data = cp.deepcopy(epochs)
    infos = [sub.info for sub in data]
    times = data[0].times

    # Checking if all times match
    for sub in data:
        if np.any(sub.times != times):
            raise Exception('Please use same time window and sampling frequency for all subjects')
    
    # Setting max jitter
    if maxjitter == 'nsamples':
        maxjitter = len(times)
    elif type(maxjitter)!=int:
        raise Exception("Please use a int as maxjitter of default 'nsamples'")

    # Fetching data
    data = [sub.get_data() for sub in data]

    # Making Surrogate
    if mode == 'trial':
        for sub in data:
            ntrials=np.shape(sub)[0]
            jitters=np.random.randint(low=minjitter,high=maxjitter,size=ntrials)
            for k in range(ntrials):
                jitter=jitters[k]
                sub[k,:,:] = np.roll(sub[k,:,:], jitter, axis=1)
    elif mode == 'subject':
        nsubs = len(data)
        jitters = np.random.randint(low=minjitter,high=maxjitter,size=nsubs)
        for i in range(len(data)):
            data[i] = np.roll(data[i], jitters[i], axis=2)
    surrogate = [mne.EpochsArray(sub, infos[i],tmin=times[0],verbose=False) for i,sub in enumerate(data)]
    return surrogate

# %% Defining Class
class gTRCA() :
    """ gTRCA Class for Group TRCA Analysis of MNE Epochs Data.
    Args:
        data (list of mne.Epochs): List of MNE Epochs to apply gTRCA.
        n_components (int or 'all', optional): Number of components to use. Defaults to 1.
        protocol_info (dict, optional): Dictionary with information about the protocol. Defaults to None.
        reg (float, optional): Regularization parameter. Defaults to 10**5.
        norm_q (bool, optional): Normalize q. Defaults to True.
        norm_y (bool, optional): Normalize y. Defaults to True.
        show_progress (bool, optional): Show progress bar. Defaults to True.
        verbose (bool, optional): Show verbose output. Defaults to True.
    """
    def __init__(self, data, n_components=1, protocol_info=None,
                  reg=10**5, norm_q=True, norm_y=True,
                  show_progress=True, verbose=True):
        self.verbose=verbose
        self.norm_y=norm_y
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
        epochs = self.group_to_array(data, norm_q)
        self.u, self.v, self.q0, self.q = self.calculate_u_q(epochs)
        self.s = self.calculate_s(epochs)
        inv_q, self.Svd = self.apply_inverse_reg(reg)
        self.rho, self.vecs = self.apply_eigen_decomposition(inv_q,reg)
        if n_components == 'all':
            n_components = len(self.rho)
        self.n_components = n_components
        self.ydata, self.maps, self.w = self.project_results(epochs, components=self.n_components, norm_y=self.norm_y)
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
        """
        This function is used to calculate the U and Q matrices.
        Args:
            epochs (list of np.ndarray): List of Subjects Data in np.ndarray format
        Returns:
            u (list of np.ndarray): List of U matrices.
            v (list of np.ndarray): List of V matrices.
            q0 (list of np.ndarray): List of Q0 matrices.
            q (np.ndarray): Q matrix as diagonally concatenated Q0 matrices.
        """
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
        """ This function is used to calculate the S matrix.
        Args:
            epochs (list of np.ndarray): List of Subjects Data in np.ndarray format
        Returns:
            s (np.ndarray): S matrix.
        """
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
        """ This function is used to regularize the S matrix. It is useful for further matrix inverse operations.
        Args:
            S (np.ndarray): S matrix.
            reg (float): Regularization parameter.
        Returns:
            reg_num (int): Number of components to be used for regularization.
        """
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
        """ This function is used to apply the inverse of the S matrix.
        Args:
            reg (float): Regularization parameter.
        Returns:
            inv_q (np.ndarray): Inverse of the S matrix.
            S (np.ndarray): S matrix.
        """
        if self.verbose:
            print('Regularizing via Ratio Eig > ('+str(reg)+')...')
        U, S, V = np.linalg.svd(self.q)
        reg_num = self.regularize_matrix(S,reg)
        inv_q = (V[0:reg_num, :].T* (1./S[0:reg_num])) @ U[:, 0:reg_num].T
        return inv_q, S

    def apply_eigen_decomposition(self, inv_q,reg):
        """ This function is used to apply the eigen decomposition of the S matrix.
        Args:
            inv_q (np.ndarray): Inverse of the S matrix.
            reg (float): Regularization parameter.
        Returns:
            rho (np.ndarray): Eigenvalues of the S matrix.
            vecs (np.ndarray): Eigenvectors of the S matrix.
        """
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

    def project_results(self, epochs=None, components=5, norm_y=True):
        """ This function is used to project the results of the S matrix.
        Args:
            epochs (list of np.ndarray): List of Subjects Data in np.ndarray format
            components (int): Number of components to be used for projection.
            norm_y (bool): Normalize the y_data.
        Returns:
            w (list of np.ndarray): List of w vectors for each subject.
            ydata (list of np.ndarray): List of y_data for each subject.
            maps (list of np.ndarray): List of maps for each subject.
            y1 (list of np.ndarray): List of y1 for each subject.
        """
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
                if norm_y:
                    for c in range(components):
                        cmean = np.mean(np.mean(ydata[i][c,:,:],axis=0)) #
                        cstd = np.std(np.mean(ydata[i][c,:,:], axis=0)) #
                        ydata[i][c,:,:] = (ydata[i][c,:,:]-cmean)/cstd #

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
        """ This function is used to plot the scalp maps.
        Args:
            component (int): Component to be plotted.
            subject (int): Subject to be plotted.
            axes (matplotlib.axes._subplots.AxesSubplot): Axes to be plotted.
        """
        if subject is None:
            mean = np.mean([self.maps[sub][component] for sub in range(self.nsubjects)], axis=0) # maps: [sub][comp](62) --> (62)
            mne.viz.plot_topomap(mean, self.mne_infos[0], axes=axes,show=False,cmap='jet');
        else:
            mne.viz.plot_topomap(self.maps[subject][component,:], self.mne_infos[subject], axes=axes,show=False,cmap='jet');
        return

    def plot_ydata(self, component=0, subject=None, axes=None, colors=['gray','red'],basecorr=None):
        """ This function is used to plot the y_data.
        Args:
            component (int): Component to be plotted.
            subject (int): Subject to be plotted.
            axes (matplotlib.axes._subplots.AxesSubplot): Axes to be plotted.
            colors (list of str): List of colors to be used.
            basecorr (list of float): List of times to be used for baseline correction.
        """
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
        """ This function is used to fix the orientation of the y_data.
            It is used to make sure that the y_data is always positive on the first peak of GMFP.
            If the previous results in a negative correlation with the mean, it is fliped.
        """
        n_components = np.shape(self.ydata[0])[0]
        ynorm = [np.zeros([n_components, self.tau])]*self.nsubjects
        for c in range(n_components):
            ynorm = [np.mean(y[c,:,:], axis=0) for y in self.ydata]
            ynorm = [(y-np.mean(y))/np.std(y) for y in ynorm]
            mean_abs = [np.abs(y) for y in ynorm]
            mean_abs = np.mean(np.array(mean_abs),axis=0)
            peak, _ = scipy.signal.find_peaks(mean_abs, distance=len(mean_abs))
            # First fix: based on first peak of GMFP
            for i in range(len(self.ydata)):
                peak_sig = np.sign(ynorm[i][peak])
                self.ydata[i][c,:,:] = peak_sig*self.ydata[i][c,:,:]
                self.maps[i][c,:] = peak_sig*self.maps[i][c,:]
                self.w[i][c,:] = peak_sig*self.w[i][c,:]
            # Second fix: based on correlation with the mean. Useful when first correction is not enough.
            # Weak components may have a negative correlation with the mean, so we flip them.
            for i in range(len(self.ydata)):
                corr = np.corrcoef(np.mean(self.ydata[i][c,:,:],axis=0),
                                   np.mean(np.array([np.mean(y[c,:,:],axis=0) for y in self.ydata]), axis=0))[0,1] # Mean of all subs ydata
                if corr < 0:
                    self.ydata[i][c,:,:] = -self.ydata[i][c,:,:]
                    self.maps[i][c,:] = -self.maps[i][c,:]
                    self.w[i][c,:] = -self.w[i][c,:]
        return

    # Needs Refactoring
    def project(self, subject, trial='all'):
        """Get predictive filter of new available data

        Args:
            subject (mne.Epochs): subject object. Please use same trial duration
                since we're using same tw_idx for selecting projection window
            trial ('all', int or list, optional): which trial to use.
                Defaults to 'all'. If list, gets form trial[0]:trial[1]

        Returns:
            w (n_components, nchannels): predictive spatial filter W(a+1)
            corrs (n_components): correlation of newsub_ydata and mean_group_ydata
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
        wg = self.w # [nsubs](n_components, nchannels)
        yg = self.ydata # [nsubs](n_components, ntrials, tau)
        yg = [np.mean(yg[sub], axis=1) for sub in range(nsubs)] # [nsubs](n_components, tau)
        yg = np.mean([yg[sub] for sub in range(nsubs)], axis=0) # [nsubs](n_components, tau)

        n_components = self.n_components
        
        # Applying Projection
        w = np.zeros([n_components, nchs])
        sub_ydata = np.zeros([n_components, tau])
        corrs_trials = np.zeros([ntrials,n_components])
        corrs_avg=np.zeros(n_components)
        maps = np.zeros([n_components, nchs])
        for c in range(n_components):
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

    # Needs Refactoring (Not that useful anymore)
    # def make_evoked(self, comp=None, subject=None):
    #     """ This function is used to make the evoked response.
    #         It can be used to make the evoked response of a single subject or the average of all subjects.
    #     Args:
    #         comp (int): Component to be used.
    #         subject (int): Subject to be used.
    #     Returns:
    #         yevoked (numpy.ndarray): mne.Evoked(...).
    #     """
    #     if comp is None:
    #         comp = self.n_components
    #     w = self.w # [subject](n_components, nchannels)
    #     ydata = self.ydata # [subject](n_components, ntrials, tau)
    #     if subject is None:
    #         subs_ymean = [np.mean(suby, axis=1) for suby in ydata]
    #         yevoked = np.zeros([self.nsubjects, self.nchannels, self.tau])
    #         for i in range(self.nsubjects):
    #             # Multiplying each subject spatial map and its respective component
    #             yevoked[i] = w[i].T @ subs_ymean[i]
    #         yevoked = np.mean(yevoked, axis=0) # Mean of Subjects
    #     else:
    #         sub_ymean = np.mean(ydata[subject], axis=1)
    #         yevoked = np.zeros([self.nchannels, self.tau])
    #         yevoked = w[subject].T @ sub_ymean
    #     yevoked = mne.EvokedArray(yevoked, self.mne_info, tmin=self.tmin) 
    #     return yevoked
