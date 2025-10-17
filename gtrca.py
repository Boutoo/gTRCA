# ---------------------------------------------------------------------------
# gTRCA for Evoked Potentials
# File: gtrca.py
# Version: 1.2.0
# Author: Couto, B.A.N. and Casali, A. G. (2025). 
# Date: 2025-03-29
# Description: Group Task Related Component Analysis (gTRCA) adapted for Evoked Potentials.
# ---------------------------------------------------------------------------

# Imports
import mne
import numpy as np
import scipy
import scipy.linalg
import scipy.signal


class gTRCA():
    """
    Group Task Related Analysis (gTRCA) class adapted for Evoked Potentials

    Please refer to: 
    Couto et al., (2025). "Extracting reproducible components from TMS-EEG recordings with Group Task-related Component Analysis"

    Usage: 
        from gtrca import gTRCA
        gtrca = gTRCA() # Inicialization
        gtrca.fit(epochs) # fit gtrca to epochs (epochs= list of MNE epochs)
        y,maps=gtrca.get_component(component=[0])# extract first component time-series and spatial maps 

    For more information, see the documentation of the fit method and
    Couto et al., (2025). "Extracting reproducible components from TMS-EEG recordings with Group Task-related Component Analysis".


    Attributes:
        times (numpy.ndarray): Time vector 
        infos (list of mne.Info): List of mne.Info objects 
        number_of_channels (int): Number of channels
        number_of_trials (int): Number of trials 
        stimuli_onset (int): Onset of the stimuli in samples.
        data (list of numpy.ndarray): Data used to fit the gTRCA object.
        evokeds (numpy.ndarray): Evoked responses (average across trials) 
        mean_trials_covariances (numpy.ndarray): Mean trials covariance matrix 
        covariance_matrices (numpy.ndarray): Covariance matrices
        s_matrix (numpy.ndarray): S matrix 
        eigenvalues (numpy.ndarray): Eigenvalues 
        eigenvectors (numpy.ndarray): Eigenvectors
    """
    def __init__(self):
        # Fitted Data
        self.times = None
        self.infos = None
        self.number_of_channels = None
        self.number_of_trials = None
        self.stimuli_onset = None
        self.data = None
        self.evokeds = None
        self.reg = None

        # Important Matrices
        self.mean_trials_covariances = None
        self.covariance_matrices = None
        self.s_matrix = None

        # gTRCA
        self.eigenvalues = None
        self.eigenvectors = None

        # Alterations
        self.shifts = None
        self.drop_subjects = []
    
    def __repr__(self) -> str:
        """Build string representation."""
        class_name = self.__class__.__name__
        repr_str = f'<{class_name} |'
        if self.data is None:
            repr_str += f'\n    Please fit data to gTRCA using .fit(data)'
        else:
            repr_str += f'\n    Number of Subjects : {len(self.data)}'
            repr_str += f'\n    Times : {self.times[0]} - {self.times[-1]}'
            repr_str += f'\n    Sampling Frequency : {1/(self.times[1]-self.times[0]):.2f} Hz'
            repr_str += f'\n    Shifted : {np.any(np.concatenate(self.shifts)!=0)}'
            repr_str += f'\n    Dropped Subjects: {self.drop_subjects}'
            repr_str += f'\n>'
        return repr_str
    
    def get_info(self):
        """ Returns a dict with some information about the gTRCA object.
        """
        info = {
            'n_subjects': len(self.data),
            'n_channels': self.number_of_channels,
            'n_trials': self.number_of_trials,
            'times': self.times,
            'sampling_frequency': 1/(self.times[1]-self.times[0]),
            'shifted': np.any(np.concatenate(self.shifts)!=0),
            'dropped_subjects': self.drop_subjects
        }
        return info
    
    def fit(self, data, onset=0,
                standardize=True,
                reg=10**5,
                verbose=True,
                progress_bar=True):
        """ Fits the gTRCA object to the data. 
        Data must be a list of mne.Epochs(...) format.

        Args:
            data (list): List of mne.Epochs(...).
            onset (float, optional) : Onset of the stimuli in seconds. Defaults to 0.
            apply_normalization (bool, optional): Whether to apply normalization to the data. Defaults to True.
            reg (int,optional): regularization parameter. Default to 10**5
            verbose (bool, optional): Whether to print out information. Defaults to False.
        """
        # Fetching Informations
        self.components_with_fixed_orientation = []
        self.infos = [sub.info for sub in data]
        self.times = data[0].times
        self.reg = reg
        self.stimuli_onset = data[0].time_as_index(onset)[0]

        # Checking if all times match
        for sub in data:
            if np.any(sub.times != self.times):
                raise Exception('Please use same time window and sampling frequency for all subjects')

        # Fetching data
        data = [sub.get_data(copy=True) for sub in data]
        self.number_of_trials = [np.shape(sub)[0] for sub in data]
        self.number_of_channels = [np.shape(sub)[1] for sub in data]

        if standardize:
            # Applying Standardization
            if verbose:
                print('Applying Standardization...')
            data = [self._apply_standardization(sub) for sub in data]

        # Saving data
        self.data = data
        self.shifts = [np.zeros(sub.shape[0], dtype=int) for sub in data]

        # Calculating Matrices and Applying gTRCA
        _ = self._calculate_matrices(reg=self.reg, verbose=verbose, progress_bar=progress_bar)

        _ = self._apply_gtrca(reg=self.reg, verbose=verbose)

        # Printing out information
        if verbose:
            print(f'✅ gTRCA fitted to {len(data)} subjects')
        return

    def get_component(self,
                      component=[0],
                      subject='all',
                      average=True,
                      normalization=True,
                      normalization_mode='subject',
                      normalization_window=None,
                      orientation=True,
                      orientation_window=None,
                      orientation_Spatialmaps=True,
                      verbose=False):
        """ Returns projected data.

        Args:
            component (list, integers): Components to return.
            subject (int): Subject to return. Default to 'all'
            average (bool): Whether to average the data across trials.
            normalization (bool): whether to normalize projections. Defaults to True.
            normalization_mode (str): Normalization mode to apply. Defaults to 'subject'. Can be 'group', 'subject', or 'none'.
            normalization_window (str): Normalization window to apply. Defaults to None.
                Can be 'baseline', None or list of two integers indicating which samples to use.
            orientation (bool): whether to align components. Defaults to True.
            orientation_window (list, time in seconds): Window to use when orienting components temporally. 
                The peak of the mean component within the time interval will be oriented in the positive direction. 
                Can be None (default) or list with two time instants indicating which samples to use (example: [0,0.1] ).
                If the interval is set to None, the positive direction will correspond to the peak computed from the entire time series.
            orientation_Spatialmaps (bool): whether to apply spatial orientation to spatial maps when aligning components. Default to True. 

        Returns:
            projections (np.ndarray): Projected data. If average is True, shape is (n_subs, n_components, n_times). Else, returns a list with each element representing a subject with sizes (n_trial, n_components, n_times).
            spatial_maps (np.ndarray): Spatial maps. Shape is (n_subs, n_components, n_channels).
        """

        projections,spatial_maps = self._calculate_proj_maps(component=component,subject=subject)

        if normalization:
            if verbose:
                print(f'- Applying Normalization...')
            projections = self._apply_components_normalization(projections,
                                                           mode=normalization_mode,
                                                           window=normalization_window)
        if orientation:
            if orientation_window is not None:
                window=np.logical_and(self.times>=orientation_window[0],self.times<=orientation_window[1])
            else:
                window=None
            if verbose:
                print(f'- Checking Orientation...')
            projections,spatial_maps = self._apply_components_orientation(projections, spatial_maps,window=window,orientation_Projections=orientation,orientation_Spatialmaps=orientation_Spatialmaps)

        if average:
            projections = np.array([np.mean(proj, axis=0) for proj in projections])

        return projections, spatial_maps



    def project(self, new_subject, component=0,
                average=True,
                normalization_window=None,
                reg=10**5,
                verbose=False):
        """ Projects new subject data to the gTRCA space. """""" Projects new subject data to the gTRCA space.
        
        Args:
            new_subject (mne.Epochs): New subject data to be projected.
            component (int, optional): Component to project the data to. Defaults to 0.
            verbose (bool, optional): Whether to print out information. Defaults to False.

        Returns:

        """

        # New subject and new subject Evoked (X_{A+1})):
        new_subject = new_subject.get_data(copy=True)
        new_subject = self._apply_standardization(new_subject)

        # New subject Covariance Matrix (Q_{A+1}):
        new_subject_cov = self._calculate_covariance_matrix(new_subject)
        new_subject_cov_inv = self._apply_inverse_regularization([new_subject_cov], reg)[0]

        # New subject filter (w_{A+1}):
        nchannels = [self.number_of_channels[i] for i in range(len(self.data)) if i not in self.drop_subjects]

        get_idx = lambda i: np.sum(nchannels[:i]) if i!=0 else 0
        get_filters = lambda i: self.eigenvectors[get_idx(i):get_idx(i+1), component]
        
        sum_of_filters_w = np.sum([(self.evokeds[i].T @ get_filters(i)) for i in range(len(self.evokeds))], axis=0)
        new_subject_w = (1/2*len(self.evokeds)) * (new_subject_cov_inv @ np.mean(new_subject, axis=0)) @ sum_of_filters_w

        # Projecting new subject (Y_{A+1}):
        new_subject_projection = [new_subject_w.T @ new_subject[trial, :, :] for trial in range(np.shape(new_subject)[0])]
        new_subject_projection = np.array(new_subject_projection)
        new_subject_spatial_map = new_subject_cov @ new_subject_w


        if verbose:
            print(f'- Applying Normalization...')
        new_subject_projection = self._apply_components_normalization([new_subject_projection],
                                                           mode='subject',
                                                           window=normalization_window)
        new_subject_projection=new_subject_projection[0]

        # Applying Reorientation
        correlation = np.corrcoef(np.mean(new_subject_projection, axis=0), self.get_average_component(component)[0])[0,1]
        if correlation < 0:
            new_subject_projection *= -1
            new_subject_spatial_map *= -1
            correlation *= -1
        
        # Averaging if necessary
        if average:
            new_subject_projection = np.mean(new_subject_projection, axis=0)

        return new_subject_projection, new_subject_spatial_map, correlation

    def run_surrogate(self, mode='subject', minjitter=0, maxjitter='nsamples'):
        """ Runs a surrogate analysis to get the higher eigenvalue obtained with gTRCA.
        
        Args:
            mode (str, optional): Surrogate mode to use. Defaults to 'subject'. Can be 'trial' or 'subject'.
            minjitter (int, optional): Minimum jitter to use. Defaults to 0.
            maxjitter (int, optional): Maximum jitter to use. Defaults to 'nsamples'.

        Returns:
            surrogate_eigenvalue (float): Higher eigenvalue obtained with gTRCA.
        """
        self._shift_data(mode=mode, minjitter=minjitter, maxjitter=maxjitter)
        _ = self._calculate_matrices(reg=self.reg, verbose=False)
        _ = self._apply_gtrca(reg=self.reg, verbose=False)
        return self.eigenvalues[0]

    def reset(self):
        """ Restores the structure to its original configuration, reverting changes as shifts or drops"""
        self._unshift_data()
        self._calculate_matrices(reg=self.reg)
        self._apply_gtrca(reg=self.reg)
        print('Done resetting changes! ✅')

    def drop(self, subject=None,verbose=True):
        """ Drops subject(s) from the gTRCA object.
        
        Args:
            subject (int or list or None): Subject(s) to drop. Defaults to None, which keep all subjects.
        """
        if subject == None:
            subject = []
        if type(subject) == int:
            subject = [subject]
        elif type(subject) != list:
            raise Exception('Please provide a list of subjects to drop.')
        self.drop_subjects = subject
        self._calculate_matrices(reg=self.reg,verbose=verbose)
        self._apply_gtrca(reg=self.reg,verbose=verbose)
        if verbose:
            print('Done running gTRCA! ✅')
            print('Subjects dropped: ', subject)
            print('Map available at .get_drop_map()')
        pass

    def get_drop_map(self):
        """ Returns the map of the dropped subjects"""
        if self.drop_subjects == None:
            raise Exception('No subjects were dropped.')
        else:
            original = range(len(self.data))
            dropped = self.drop_subjects
            drop_map = [i for i in original if i not in dropped]
            return drop_map

    def _calculate_matrices(self, reg, verbose=True, progress_bar=True):
        trial_duration = len(self.times)

        drop_subs = lambda x: [x[i] for i in range(len(x)) if i not in self.drop_subjects]
        data = drop_subs(self.data) # Removing dropped subjects

        # Making Evokeds (Uα)
        if verbose:
            print('Making Evokeds...')
            if progress_bar:
                self._print_progress_bar(0, len(data), prefix='U\u03B1:', suffix='Complete')

        self.evokeds = []
        for s, sub in enumerate(data):
            if verbose and progress_bar:
                self._print_progress_bar(s+1, len(data), prefix='U\u03B1:', suffix='Complete')
            self.evokeds.append(np.mean(sub, axis=0))

        # Calculating Mean Covariance Matrix (Vα)
        if verbose:
            print('Calculating Mean Covariance Matrix...')
            if progress_bar:
                self._print_progress_bar(0, len(data), prefix='V\u03B1:', suffix='Complete')
        
        self.mean_trials_covariances = []
        for s, sub in enumerate(data):
            if verbose and progress_bar:
                self._print_progress_bar(s+1, len(data), prefix='V\u03B1:', suffix='Complete')
            self.mean_trials_covariances.append(self._calculate_mean_trials_covariance(sub))

        # Calculating Covariance Matrices (Qα)
        if verbose:
            print('Calculating Covariance Matrices...')
            if progress_bar:
                self._print_progress_bar(0, len(data), prefix='Q\u03B1:', suffix='Complete')
        
        self.covariance_matrices = []
        for s, sub in enumerate(data):
            if verbose and progress_bar:
                self._print_progress_bar(s+1, len(data), prefix='Q\u03B1:', suffix='Complete')
            self.covariance_matrices.append(self._calculate_covariance_matrix(sub))

        # Calculating S Matrix
        if verbose:
            print('Calculating S Matrix...')
        self.s_matrix = self._calculate_s_matrix(
            self.evokeds,
            self.mean_trials_covariances,
            trial_duration,
            drop_subs(self.number_of_trials), # Removing dropped subjects
            drop_subs(self.number_of_channels), # Removing dropped subjects
            verbose=verbose,
            progress_bar=progress_bar
            )
        
        return # (U,V,Q,S)

    def _apply_gtrca(self, reg, verbose=True):

        # Applying Inverse Regularization
        if verbose:
            print('Applying Inverse Regularization...')
        covariances_inverse, _ = self._apply_inverse_regularization(self.covariance_matrices, reg, verbose=verbose)

        # Find Eigenvalues
        if verbose:
            print('Solving Eigenvalues and Eigenvectors...')
        self.eigenvalues, self.eigenvectors = self._apply_eigen_decomposition(covariances_inverse @ self.s_matrix, reg, verbose=verbose)

        return # Eigenvals, Eigenvectors

    def _calculate_mean_trials_covariance(self, subject):
        """ Calculates the mean covariance matrix of the data.

        Args:
            data (np.ndarray): Data to calculate the mean covariance matrix of (trials, channels, samples).

        Returns:
            mean_trials_covariances (np.ndarray): Mean covariance matrix of the data.
        """
        ntrials = np.shape(subject)[0]
        mean_trials_covariance = np.sum([subject[trial,:,:] @ subject[trial,:,:].T for trial in range(ntrials)], axis=0)
        mean_trials_covariance = mean_trials_covariance/ntrials
        return mean_trials_covariance

    def _calculate_covariance_matrix(self, subject):
        """ Concatenates Trials and calculates the covariance matrices of the data.

        Args:
            data (np.array): List of arrays to calculate the covariance matrices of. Arrays must be (trials, channels, samples).

        Returns:
            covariance_matrices (list): List of covariance matrices of the data.
        """
        continuous = self._build_continuous(subject)
        covariance_matrix = (1/np.shape(continuous)[1])*(continuous @ continuous.T)
        return covariance_matrix

    def _calculate_s_matrix(self,
                            evokeds,
                            mean_trials_covariance,
                            trial_duration,
                            number_of_trials,
                            number_of_channels,
                            verbose=False,
                            progress_bar=False):
        """ Calculates the S Matrix.

        Args:
            evokeds (list): List of evokeds of the data.
            mean_trials_covariance (list): List of mean trials covariance matrices of the data.
            trial_duration (int): Duration of each trial in samples.
            number_of_trials (list): List of number of trials of the data.
            number_of_channels (list): List of number of channels of the data.
            verbose (bool, optional): Whether to print out information. Defaults to False.
        """
        s_size = np.sum(number_of_channels)
        s_matrix = np.zeros((s_size, s_size))
        get_idx = lambda i: np.sum(number_of_channels[:i]) if i!=0 else 0
        if verbose and progress_bar:
                self._print_progress_bar(0, len(evokeds), prefix='S\u03B1\u03B2:', suffix='Complete')
        for a in range(len(evokeds)):
            if verbose and progress_bar:
                self._print_progress_bar(a+1, len(evokeds), prefix='S\u03B1\u03B2:', suffix='Complete')
            for b in range(len(evokeds)):
                if a == b:
                    s_matrix[
                        get_idx(a):get_idx(a+1),
                        get_idx(b):get_idx(b+1)
                        ] = 2*(number_of_trials[a]/((number_of_trials[a]-1)*trial_duration)) * \
                            ((evokeds[a] @ evokeds[a].T) - \
                              (1/number_of_trials[a])*mean_trials_covariance[a])
                else:
                    s_matrix[
                        get_idx(a):get_idx(a+1),
                        get_idx(b):get_idx(b+1)
                    ] = (1/trial_duration) * (evokeds[a]@evokeds[b].T)
        
        return s_matrix

    def _apply_standardization(self, subject):
        """ Standardizes data so that channels get zero mean and unit std.

        Args:
            data (np.ndarray): Data to apply normalization to.

        Returns:
            data (np.ndarray): Normalized data.
        """
        # Reshape data to have shape (channels, trials*time)
        subject_reshaped = np.transpose(subject, (1, 0, 2)).reshape(subject.shape[1], -1)

        # Compute mean and std
        mean = np.mean(subject_reshaped, axis=1, keepdims=True)
        std = np.std(subject_reshaped, axis=1, keepdims=True)

        # Standardize the data
        subject_standardized = (subject_reshaped - mean) / std

        # Reshape back to the original shape
        subject_standardized = subject_standardized.reshape(subject.shape[1], subject.shape[0], subject.shape[2])

        # Transpose back to original order
        subject_standardized = np.transpose(subject_standardized, (1, 0, 2))

        return subject_standardized

    def _build_continuous(self, subject):
        # Reshape data to have shape (channels, trials*time)
        subject_reshaped = np.transpose(subject, (1, 0, 2)).reshape(subject.shape[1], -1)
        return subject_reshaped

    def _apply_inverse_regularization(self, data, reg, verbose=False):
        """ This function regularizes a given matrix and calculates it's inverse.
        Args:
            data (np.array): List of matrix to be regularized and inverted.
            reg (float): Regularization parameter.
        Returns:
            data_inverse (np.ndarray): Inverse of the Q matrix.
            S (np.ndarray): S from SVD matrix.
        """
        data = scipy.linalg.block_diag(*data)
        if verbose:
            print('Regularizing via Ratio Eig > ('+str(reg)+')...')

        U, S, V = np.linalg.svd(data)
        reg_num = self._apply_regularization(S,reg)
        data_inverse = (V[0:reg_num, :].T* (1./S[0:reg_num])) @ U[:, 0:reg_num].T
        return data_inverse, S
    
    def _apply_regularization(self, S, reg):
        """ This function is used to regularize the S matrix (from SVD).
        It is useful for further matrix inverse operations.
        Args:
            S (np.ndarray): S matrix.
            reg (float): Regularization parameter.
        Returns:
            reg_num (int): Number of components to be used for regularization.
        """

        eps = np.finfo(float).eps
        ix1 = np.where(np.abs(S)<1000*np.finfo(float).eps)[0] # Removing null components
        ix2 = np.where((S[0:-1]/(eps+S[1:]))>reg)[0] # Cut-off based on eingenvalue ratios
        ix = np.union1d(ix1,ix2)
        if len(ix)==0:
            reg_num=len(S)
        else:
            reg_num=np.min(ix)
        return reg_num

    def _apply_eigen_decomposition(self, data, reg, verbose=False):
        """This function applies eigen decomposition on given data.
            It also sort values in descending order and aplies regularization.

        Args:
            data (np.array): List of matrix to be regularized and inverted.
            reg (float): Regularization parameter.

        Returns:

        """
        eigenvalues, eigenvectors = np.linalg.eig(data)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:,idx]
        reg_num = self._apply_regularization(abs(eigenvalues), reg)
        eigenvalues = eigenvalues[0:reg_num]
        eigenvectors = eigenvectors[:,0:reg_num]
        if np.max(abs(np.imag(eigenvalues))) != 0:
            print('Warning: There are imaginary parts in eigenvalues.')
        return eigenvalues.real, eigenvectors.real


    def _calculate_proj_maps(self,component=[0],subject='all'):

        drop_subs = lambda x: [x[i] for i in range(len(x)) if i not in self.drop_subjects]
        data = drop_subs(self.data) # Removing dropped subjects
        if subject == 'all':
            subjects = range(len(data))
        else:
            subjects = [subject]        
        number_of_channels = drop_subs(self.number_of_channels) # Removing dropped subjects
        get_idx = lambda i: np.sum(number_of_channels[:i]) if i!=0 else 0

        projections = []
        spatial_maps=[]
        for s in subjects:
            w = self.eigenvectors[get_idx(s):get_idx(s+1), component]
            projection = [w.T @ data[s][trial,:,:] for trial in range(np.shape(data[s])[0])]
            projection = np.array(projection)

            # Creating Spatial Map
            continuous_component = self._build_continuous(projection)
            covcomponent = (1/np.shape(continuous_component)[1])*(continuous_component @ continuous_component.T)
            covcomponent_inv,_=self._apply_inverse_regularization([covcomponent], 10**5, verbose=False)
            spatial_map = self.covariance_matrices[s] @ w @ covcomponent_inv

            projections.append(projection)
            spatial_maps.append(spatial_map.T)      

        return projections, spatial_maps


    def _apply_components_orientation(self,
                                        projections,
                                        spatial_maps,
                                        window=None,
                                        orientation_Projections=True,
                                        orientation_Spatialmaps=True):
        """ This function reorientates the components and spatial maps based on temporal and spatial correlations, respectively.

        Args:
            projections (np.ndarray): Projections to be reorientated.
            spatial_maps (np.ndarray): Spatial maps to be reorientated.
            window (np.ndarray): Normalization window to apply. Defaults to None.
                Can be  None or a logical array indicating which samples to use.
            orientation_Projections (bool,optional): Wheter to align projections
            orientation_Spatialmaps (bool,optional): Wheter to align spatial maps
            verbose (bool, optional): Whether to print out information. Defaults to False.

        Returns:
            projections (np.ndarray): Reorientated projections.
            spatial_maps (np.ndarray): Reorientated spatial maps.
        """


        get_evoked = lambda a: np.mean(a, axis=0)
        ncomps=np.size(projections[0],axis=1)

        # Temporal orientation:
        if orientation_Projections:
            if window is not None:
                mapa=[get_evoked(sub[:,:,window]) for sub in projections]
            else:
                mapa=[get_evoked(sub) for sub in projections]
            gmfp = np.mean([abs(x) for x in mapa], axis=0)
            gmfp_peaks=[s[0] for s in [scipy.signal.find_peaks(s, distance=len(s)) for s in gmfp]]
            for c in range(ncomps):
                for i, x in enumerate(mapa):
                    if x[c,gmfp_peaks[c]] < 0:
                        projections[i][:,c,:] *= -1
            mapa = [get_evoked(sub) for sub in projections]
            mean_component = np.mean(mapa, axis=0)
            for c in range(ncomps):
                for i, x in enumerate(mapa):
                    if np.corrcoef(x[c,:], mean_component[c,:])[0,1]<0:
                        projections[i][:,c,:] *= -1
                        spatial_maps[i][c,:] *= -1

        #Spatial orientation:
        if orientation_Spatialmaps:
            gmfp = np.mean([abs(x) for x in spatial_maps], axis=0)
            gmfp_peaks=[s[0] for s in [scipy.signal.find_peaks(s, distance=len(s)) for s in gmfp]]
            for c in range(ncomps):
                for i, x in enumerate(spatial_maps):
                    if x[c,gmfp_peaks[c]] < 0:
                        spatial_maps[i][c,:] *= -1
            mean_component = np.mean([topo for topo in spatial_maps],axis=0)
            for c in range(ncomps):
                for i, x in enumerate(spatial_maps):
                    if np.corrcoef(x[c,:], mean_component[c,:])[0,1]<0:
                        spatial_maps[i][c,:] *= -1

        return projections, spatial_maps

    def _apply_components_normalization(self, data,
                                        mode='subject',
                                        window=None):
            """ Applies component normalization.

            Args:
                data (np.ndarray): Data to apply normalization to.
                mode (str, optional): Whether to apply normalization on 'group' or 'subject' level. Defaults to 'subject'.
                window (str, optional): Window to apply normalization on, in samples. Defaults to None. If None, applies on whole data.
                    Can be 'baseline', None or list of two integers indicating which samples to use.
            
            Returns:
                data (np.ndarray): Normalized data.
            """
            get_evoked = lambda a: np.mean(a, axis=0)
            
            if mode == 'group':
                # Group Average
                group_average = np.mean(np.array([get_evoked(projection) for projection in data]), axis=0)
                
                # Group Standard Deviation on chosen window
                if window == 'baseline':
                    group_average = group_average[:,:self.stimuli_onset]
                elif window is None:
                    pass
                else:
                    group_average = group_average[:,window[0]:window[1]]

                # Normalization
                group_std = np.std(group_average,axis=1)
                for i, projection in enumerate(data):
                    for j, trial in enumerate(projection):
                        data[i][j,:,:] = [((s - np.mean(s[:self.stimuli_onset])) / group_std[k]) for k,s in enumerate(trial)]

            elif mode == 'subject':
                for i, projection in enumerate(data):
                    # Subject Average
                    subject_average = get_evoked(projection)

                    # Subject Standard Deviation on chosen window
                    if window == 'baseline':
                        subject_average = subject_average[:,:self.stimuli_onset]
                    elif window is None:
                        pass
                    else:
                        subject_average = subject_average[:,window[0]:window[1]]
                    subject_std = np.std(subject_average,axis=1)
                    
                    # Normalization
                    for j, trial in enumerate(projection):
                        data[i][j,:,:] = [((s - np.mean(s[:self.stimuli_onset])) / subject_std[k]) for k,s in enumerate(trial)]                        

            # No normalization 
            elif mode == 'none':
                pass
            else:
                raise Exception('Please choose a valid normalization method.')
            return data

    def _print_progress_bar(self, iteration, total, prefix = '', suffix = '', decimals = 1, length = 15, fill = '█'):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
        # Print New Line on Complete
        if iteration == total: 
            print()

    def _shift_data(self, mode='trial', minjitter=0, maxjitter='nsamples'):
        """ Shifts data trial or subject-wise. Useful when running surrogates.

        Args:
            mode (str): 'trial' or 'subject'
            minjitter (int): Minimum value of shift, in samples. Defaults to 0.
            maxjitter (int or 'nsamples'): Maximum value of shift, in samples. Defaults to 'nsamples', using time length.
        """

        if self.data is None:
            raise Exception('Please fit data to gTRCA before shifting.')
        
        # Setting max jitter
        if maxjitter == 'nsamples':
            maxjitter = len(self.times)
        elif type(maxjitter)!=int:
            raise Exception("Please use a int as maxjitter or default 'nsamples'")

        # Trial-based Shifting
        if mode == 'trial':
            for i, sub in enumerate(self.data):
                ntrials = sub.shape[0]
                jitters = np.random.randint(low=minjitter, high=maxjitter, size=ntrials, dtype=int)
                self.shifts[i] += jitters
                rolled_sub = np.zeros_like(sub)
                for k in range(ntrials):
                    jitter=jitters[k]
                    rolled_sub[k,:,:] = np.roll(sub[k,:,:], jitter, axis=1)
                self.data[i] = rolled_sub
    
        # Subject-based Shifting
        elif mode == 'subject':
            nsubs = len(self.data)
            jitters = np.random.randint(low=minjitter, high=maxjitter, size=nsubs, dtype=int)
            for i, sub in enumerate(self.data):
                rolled_sub = np.roll(sub, jitters[i], axis=2)
                self.shifts[i] += np.ones(sub.shape[0], dtype=int)+jitters[i] # Since every trial was shifted by the same ammount
                self.data[i] = rolled_sub
        pass
    
    def _unshift_data(self):
        """ Unshifts data
        """
        for i, sub in enumerate(self.data):
            ntrials = sub.shape[0]
            unshift_sub = np.zeros_like(sub)
            for k in range(ntrials):
                unshift_sub[k,:,:] = np.roll(sub[k,:,:], -self.shifts[i][k], axis=1)
            self.data[i] = unshift_sub
        self.shifts = [np.zeros(sub.shape[0], dtype=int) for sub in self.data]
        pass
