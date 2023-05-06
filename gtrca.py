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

# # Compute correlations of ymatrix
# def surr_gtrca(epochs, reg=10**5, norm_q=True):
#     # Defining Functions
#     def calculate_u_q(epochs):
#         """
#         This function is used to calculate the U and Q matrices.
#         Args:
#             epochs (list of np.ndarray): List of Subjects Data in np.ndarray format
#         Returns:
#             u (list of np.ndarray): List of U matrices.
#             v (list of np.ndarray): List of V matrices.
#             q0 (list of np.ndarray): List of Q0 matrices.
#             q (np.ndarray): Q matrix as diagonally concatenated Q0 matrices.
#         """
#         # Epochs (ntrials, nchs, tau)
#         u = []
#         v = []
#         q0 = []
#         q = np.zeros([np.sum(self.nchannels), np.sum(self.nchannels)])
#         if self.verbose:
#             print('Calculating U Matrix...')
#         for i, ep in enumerate(epochs):
#             ep = np.array(ep) # Retrieving from memory
#             if self.show_progress is True:
#                 print_progress_bar(i, self.nsubjects-1)
#             raw = np.transpose(ep, (1,0,2)) # raw = channel x time x trial
#             raw = raw.reshape(np.shape(raw)[0], -1) # concatenating all trials, raw = channel x time
#             q0 += [(raw @ raw.T)/np.shape(raw)[1]]
#             u += [np.mean(ep, axis=0)]
#             ntrials = np.shape(ep)[0] # Searching for local Ntrials size
#             vsum = 0
#             for k in range(ntrials):
#                 vsum += (ep[k,:,:] @ ep[k,:,:].T)
#             v += [vsum/ntrials]
#         if self.verbose:
#             print('Calculating Q Matrix...')
#         for n in range(self.nsubjects):
#             if self.show_progress is True:
#                 print_progress_bar(n, self.nsubjects-1)
#             q[n*self.nchannels[n]:(n+1)*self.nchannels[n],
#               n*self.nchannels[n]:(n+1)*self.nchannels[n]] = q0[n]
#         return u, v, q0, q
#     def calculate_s(epochs):
#         """ This function is used to calculate the S matrix.
#         Args:
#             epochs (list of np.ndarray): List of Subjects Data in np.ndarray format
#         Returns:
#             s (np.ndarray): S matrix.
#         """
#         # computation of S and Q matrices:
#         s = np.zeros([np.sum(nchannels), np.sum(nchannels)])
#         count = 0
#         for a in range(nsubjects):
#             for b in range(nsubjects):
#                 count += 1
#                 if a==b: # Diagonal
#                     ntrials = np.shape(epochs[a])[0]
#                     stmp = 2*(ntrials/((ntrials-1)*nsamples))* \
#                         ((u[a] @ u[a].T) - (v[a]/ntrials))
#                 else: # Off-Diagonal
#                     stmp = 1/nsamples*(u[a] @ u[b].T)
#                 s[a*nchannels[a]:(a+1)*nchannels[a],
#                   b*nchannels[b]:(b+1)*nchannels[b]] = stmp
#         return s
#     def apply_inverse_reg(q,reg):
#         """ This function is used to apply the inverse of the Q matrix.
#         Args:
#             reg (float): Regularization parameter.
#         Returns:
#             inv_q (np.ndarray): Inverse of the Q matrix.
#             S (np.ndarray): S from SVD matrix.
#         """
#         U, S, V = np.linalg.svd(q)
#         reg_num = regularize_matrix(S,reg)
#         inv_q = (V[0:reg_num, :].T* (1./S[0:reg_num])) @ U[:, 0:reg_num].T
#         return inv_q, S
#     def apply_eigen_decomposition(inv_q,s,reg):
#         """ This function is used to apply the eigen decomposition of the S matrix.
#         Args:
#             inv_q (np.ndarray): Inverse of the Q matrix.
#             reg (float): Regularization parameter.
#         Returns:
#             rho (np.ndarray): Eigenvalues of the S matrix.
#             vecs (np.ndarray): Eigenvectors of the S matrix.
#         """
#         M = inv_q @ s
#         rho, vecs = np.linalg.eig(M) 
#         indx=np.argsort(np.real(rho));indx=indx[-1::-1]
#         rho=rho[indx]
#         vecs=vecs[:,indx]
#         reg_num=regularize_matrix(np.abs(rho),reg)
#         rho=rho[0:reg_num]
#         vecs=vecs[:,0:reg_num]
#         if np.max(abs(np.imag(rho)))!=0:
#             raise NameError("Rho has complex values: check")
#         rho=np.real(rho)
#         vecs=np.real(vecs)
#         return rho, vecs
#     def regularize_matrix(S,reg):
#         """ This function is used to regularize the S matrix. It is useful for further matrix inverse operations.
#         Args:
#             S (np.ndarray): S matrix.
#             reg (float): Regularization parameter.
#         Returns:
#             reg_num (int): Number of components to be used for regularization.
#         """
#         eps=np.finfo(float).eps
#         ix1=np.where(np.abs(S)<1000*np.finfo(float).eps)[0] #removing null components
#         ix2 = np.where((S[0:-1]/(eps+S[1:]))>reg)[0] #cut-off based on eingenvalue ratios
#         ix=np.union1d(ix1,ix2)
#         if len(ix)==0:
#             reg_num=len(S)
#         else:
#             reg_num=np.min(ix)
#         return reg_num
    
#     # Checking input
#     nsubjects = len(epochs)
#     nchannels = np.array([len(sub.info['ch_names']) for sub in epochs])
#     times = epochs[0].times # If Group: assume all subjects have the same times
#     for epoch in epochs:
#         if np.any(epoch.times != times):
#             raise Exception('Please use Epochs with same Time Window and sampling frequency.')
#     tmin, tmax = times[0], times[-1]
#     nsamples = len(times)
    
#     print(epochs)

#     # Q normalization
#     if norm_q:
#         norm_epochs=[]
#         for i, epoch in enumerate(epochs):
#             epoch = epoch.get_data()
#             ntrials, nchs, tau = np.shape(epoch) # (ntrials, nchs, tau)
#             for ch in range(nchs):
#                 epoch[:,ch,:] = (epoch[:,ch,:]-np.mean(epoch[:,ch,:]))/np.std(epoch[:,ch,:])
#             epochs.append(epoch)
#     else:
#         epochs = [epoch.get_data() for epoch in epochs] # Get Epoches Data

#     # Calculating
#     u, v, q0, q = calculate_u_q(epochs)
#     s = calculate_s(epochs)
#     inv_q, _ = apply_inverse_reg(q, reg)
#     eigenvalues, _ = apply_eigen_decomposition(inv_q, s,reg)
#     return eigenvalues[0]

# %% Defining Class
class gTRCA():
    """ gTRCA Class for Group TRCA Analysis of MNE Epochs Data.
    Args:
        data (list of mne.Epochs): List of MNE Epochs to apply gTRCA.
        protocol_info (dict, optional): Dictionary with information about the protocol. Defaults to None.
        reg (float, optional): Regularization parameter. Defaults to 10**5.
        norm_q (bool, optional): Normalize q. Defaults to True.
        norm_y (bool, optional): Normalize y. Defaults to True.
        show_progress (bool, optional): Show progress bar. Defaults to True.
        verbose (bool, optional): Show verbose output. Defaults to True.
    """
    def __init__(self, data, evt_onset=0, protocol_info=None,
                  reg=10**5, norm_q=True, norm_y=True,
                  show_progress=True, verbose=True):
        self.protocol_info = protocol_info
        if verbose == False:
            show_progress = False
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
        self.evt_onset = int(data[0].time_as_index(evt_onset))
        self.tmin, self.tmax = self.times[0], self.times[-1]
        self.nsamples = len(self.times)
        self.epochs_data = self.group_to_array(data, norm_q)
        self.u, self.v, self.q0, self.q = self.calculate_u_q(self.epochs_data)
        self.s = self.calculate_s(self.epochs_data)
        inv_q, self.Svd = self.apply_inverse_reg(reg)
        self.eigenvalues, self.eigenvectors = self.apply_eigen_decomposition(inv_q,reg)

    def proj(self, n_components=1, fix_orientation=True):
        """proj
        This function is used to project the data into the gTRCA space.
        Args:
            n_components (int): Number of components to project to.

        Returns:
            ydata (list of np.ndarray): List of Subjects Data projected into gTRCA space.
            [(trials, n_components, tau),...,(n_components, trial, tau)]
            maps (list of np.ndarray): List of Subjects Maps.
            w (np.ndarray): Weights of the gTRCA components.
        """
        ydata, maps, w = self.project_results(self.epochs_data, components=n_components, norm_y=self.norm_y)
        if fix_orientation:
            ydata, maps, w = self.fix_ydata_orientation(ydata, maps, w)

        # Saving results
        self.ydata = ydata
        self.maps = maps
        self.w = w
        return ydata, maps, w

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
                    stmp = 2*(ntrials/((ntrials-1)*self.nsamples))* \
                        ((self.u[a] @ self.u[a].T) - (self.v[a]/ntrials))
                else: # Off-Diagonal
                    stmp = 1/self.nsamples*(self.u[a] @ self.u[b].T)
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
        """ This function is used to apply the inverse of the Q matrix.
        Args:
            reg (float): Regularization parameter.
        Returns:
            inv_q (np.ndarray): Inverse of the Q matrix.
            S (np.ndarray): S from SVD matrix.
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
        tau = self.nsamples
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
                    w[i][c,:] = self.eigenvectors[i*nchs:(i+1)*nchs, c]
                
                # Building y_data
                ydata[i] = np.zeros([components, ntrials, tau])
                for c in range(components):
                    for k in range(ntrials):
                        ydata[i][c,k,:] = w[i][c,:].T @ ep[k,:,:]
                
                # Normalization
                if norm_y:
                    for c in range(components):
                        cmean = np.mean(np.mean(ydata[i][c,:,:self.evt_onset],axis=0)) # Evoked Baseline Mean
                        cstd = np.std(np.mean(ydata[i][c,:,:], axis=0)) # Evoked Std
                        ydata[i][c,:,:] = (ydata[i][c,:,:]-cmean)/cstd

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

    def fix_ydata_orientation(self, ydata, maps, w):
        """ This function is used to fix the orientation of the y_data.
            It is used to make sure that the y_data is always positive on the first peak of GMFP.
            If the previous results in a negative correlation with the mean, it is fliped.
        """
        n_components = np.shape(ydata[0])[0]
        ynorm = [np.zeros([n_components, self.nsamples])]*self.nsubjects
        for c in range(n_components):
            ynorm = [np.mean(y[c,:,:], axis=0) for y in ydata]
            ynorm = [(y-np.mean(y))/np.std(y) for y in ynorm]
            mean_abs = [np.abs(y) for y in ynorm]
            mean_abs = np.mean(np.array(mean_abs),axis=0)
            peak, _ = scipy.signal.find_peaks(mean_abs, distance=len(mean_abs))
            # First fix: based on first peak of GMFP
            for i in range(len(ydata)):
                peak_sig = np.sign(ynorm[i][peak])
                ydata[i][c,:,:] = peak_sig*ydata[i][c,:,:]
                maps[i][c,:] = peak_sig*maps[i][c,:]
                w[i][c,:] = peak_sig*w[i][c,:]
            # Second fix: based on correlation with the mean. Useful when first correction is not enough.
            # Weak components may have a negative correlation with the mean, so we flip them.
            for i in range(len(ydata)):
                corr = np.corrcoef(np.mean(ydata[i][c,:,:],axis=0),
                                   np.mean(np.array([np.mean(y[c,:,:],axis=0) for y in ydata]), axis=0))[0,1] # Mean of all subs ydata
                if corr < 0:
                    ydata[i][c,:,:] = -ydata[i][c,:,:]
                    maps[i][c,:] = -maps[i][c,:]
                    w[i][c,:] = -w[i][c,:]
        return ydata, maps, w

    def fit(self, new_epoch, return_correlation=False,
            component=0, weights=None, reg=10**5
            correlation_times=[None,None]):
        """ This function is used to fit a new subject to the model.
        Args:
            new_epoch (mne.Epochs): New subject data in mne.Epochs format.
            component (int): Component to be used for fitting.
            return_correlation (bool): Return only the correlation between the new subject and the model. Defaults to False.
            weights (np.ndarray): Weights to be used for fitting. If None, the weights from the last projection will be used.
        Returns:
            y (np.ndarray): y for the new subject.
            w (np.ndarray): w for the new subject.
            map (np.ndarray): map for the new subject.
        """

        try:
            if weights == None:
                weights = self.w
        except:
            raise Exception('No weights provided. Please make a projection first or provide weights.')

        # Get new subject data
        new_sub = new_epoch.get_data()
        ntrials, nchannels, nsamples = np.shape(new_sub)

        # Covariance Matrix
        continuous = np.concatenate([new_sub[i,:,:] for i in range(ntrials)], axis=1)
        q_new = (1/nsamples)*(continuous@continuous.T)

        # Making inverse of q matrix with regularization
        U, S, V = np.linalg.svd(q_new)
        reg_num = self.regularize_matrix(S, reg)
        inv_q_new = (V[0:reg_num, :].T* (1./S[0:reg_num])) @ U[:, 0:reg_num].T

        # Weights
        mean_weight = np.mean(weights, axis=0)
        sum_weights = [self.u[i].T @ weights[i][component,:] for i in range(self.nsubjects)]
        sum_weights = np.sum(sum_weights, axis=0)

        # New Weights
        w = (1/2*self.nsubjects) * (inv_q_new @ np.mean(new_sub, axis=0)) @ sum_weights

        y = np.array([w.T @ new_sub[i,:,:] for i in range(ntrials)])
        y = y/np.std(np.mean(y, axis=0)) # Normalization

        map = q_new @ w

        # Fixing Polarity
        mean_ydata = np.mean([np.mean(ydata[component,:,:], axis=0) for ydata in self.ydata], axis=0)
        mean_map = np.mean([map[component, :] for map in self.maps], axis=0)

        # Chosing times for correlation: [tc0, tc1]
        time_corr = np.corrcoef(np.mean(y, axis=0), mean_ydata)[0,1]
        if time_corr < 0:
            y = -y
            map = -map
            time_corr = -time_corr
        
        if return_correlation:
            map_corr = np.corrcoef(map, mean_map)[0,1]
            return time_corr, map_corr
        else:
            return y, map, w

# ADD:
# gtrca_surr() -> minimal gtrca, only for extracting first eigenvalue
# Check if there's work about classification on MMN
