# ---------------------------------------------------------------------------
# TRCA and gTRCA for Evoked Potentials
# File: gtrca.py
# Version: 1.0.0
# Author: Couto, B.A.N. (2023). Member of the Neuroengineering Lab from the Federal University of São Paulo.
# Date: 2023-06-06
# Description: Task Related Component Analysis (TRCA) and Group Task Related Component Analysis (gTRCA) adapted for Evoked Potentials.
# ---------------------------------------------------------------------------

# Imports
import mne
import numpy as np
import scipy
import scipy.linalg
import scipy.signal

import copy as cp

# TRCA
class TRCA():
    """ Task Related Component Analysis (TRCA) is a method for extracting maximally reproducible components within a subject.
    
    Implemented according to the paper:
    Tanaka, H. (2020). Group task-related component analysis (gTRCA): A multivariate method for inter-trial reproducibility and inter-subject similarity maximization for EEG data analysis. Scientific Reports, 10(1), 84. https://doi.org/10.1038/s41598-019-56962-2

    But adapted for Evoked Potentials that are already segmented and with a 'subject' surrogate method that allows for a more robust statistical analysis for groups.

    This Python implementation was made by Couto, B.A.N. (2023)
    Member of the Neuroengineering Lab from the Federal University of São Paulo.

    For more information, see the documentation of the fit method.

    Attributes:
        S (np.ndarray): S matrix as in paper.
        Q (np.ndarray): Covariance Matrix as in paper.
        U (np.ndarray): Evoked Reponse.
        V (np.ndarray): Mean Covariance Matrix (trial-wise).
        eigenvectors (np.ndarray): Eigenvectors of the (Q^-1@S) matrix.
        eigenvalues (np.ndarray): Eigenvalues of the (Q^-1@S) matrix.
        data (np.ndarray): Data used for fitting.
        info (mne.Info): Info from the epochs.
    """

    def __init__(self):
        self.S = None
        self.Q = None
        self.U = None
        self.V = None
        self.eigenvectors = None
        self.eigenvalues = None
        self.data = None
        self.info = None
    
    def fit(self, epochs, reg=10**5, verbose=True):
        """ Fits the TRCA model to the given epochs.
        Args:
            epochs (mne.Epochs): Epochs to be fitted.
            reg (float): Regularization parameter.
        """

        self.info = epochs.info
        data = epochs.get_data()
        # Normalize each channel to have zero mean and unit variance
        data = self._apply_normalization(data)
        self.data = data
        self.Q = self._calculate_Q(data)
        self.U = self._calculate_U(data)
        self.V = self._calculate_V(data)
        self.S = self._calculate_S(data, self.U, self.V)
        inv_q, _ = self._apply_inverse_regularization(self.Q, reg, verbose=verbose)
        eigenvalues, eigenvectors = np.linalg.eig(inv_q @ self.S)

        # Sort eigenvectors by eigenvalues
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:,idx]

        self.eigenvectors = eigenvectors
        self.eigenvalues = eigenvalues

        if verbose:
            print(' Done fitting TRCA! ✅')

    def get_components(self, component='all', average=True):
        """ Returns the components of the fitted TRCA model.
        Args:
            component (int): Component to be returned. If 'all', returns all components.
            average (bool): If True, returns the average of all trials.
        Returns:
            y (np.ndarray): Component(s) of the fitted TRCA model.
            maps (np.ndarray): Spatial maps of the component(s).
        """

        if component == 'all':
            y = self.eigenvectors.T @ self.data
            mean_data = np.mean(self.data, axis=0)
            mean_data = abs(mean_data)
            mean_data = np.sum(mean_data, axis=0)
            for c in range(y.shape[1]):
                avg_y = np.mean(y[:,c,:], axis=0)
                sign = np.sign(np.corrcoef(avg_y, mean_data)[0, 1])
                y[c,:] = sign * y[c,:]
                self.eigenvectors[:,c] = sign * self.eigenvectors[:,c]
            maps = self.Q @ self.eigenvectors
        else:
            y = self.eigenvectors[:,component].T @ self.data
            mean_data = np.mean(self.data, axis=0)
            mean_data = abs(mean_data)
            mean_data = np.sum(mean_data, axis=0)
            avg_y = np.mean(y, axis=0)
            sign = np.sign(np.corrcoef(avg_y, mean_data)[0, 1])
            y = sign * y
            self.eigenvectors[:,component] = sign * self.eigenvectors[:,component]
            maps = self.Q @ self.eigenvectors[:,component]
        if average:
            y = np.mean(y, axis=0)
        return y, maps

    def run_surrogate(self, min_jitter=0, max_jitter='nsamples', verbose=False):
        """ Runs a surrogate analysis on the fitted TRCA model.
        Args:
            min_jitter (int): Minimum jitter to be applied to the data.
            max_jitter (int): Maximum jitter to be applied to the data. If 'nsamples', max_jitter is set to the number of samples.
        Returns:
            eigenvalue (np.ndarray): First eigenvalue from surrogate TRCA.
        """

        surrogate = self.data.copy()
        ntrials, nchannels, nsamples = surrogate.shape
        if max_jitter == 'nsamples':
            max_jitter = nsamples
        
        for trial in range(ntrials):
            jitter = np.random.randint(min_jitter, max_jitter)
            surrogate[trial,:,:] = np.roll(surrogate[trial,:,:], jitter, axis=1)

        surrogate = mne.EpochsArray(surrogate, self.info, verbose=False)
        surr_trca = TRCA()
        surr_trca.fit(surrogate, verbose=verbose)
        return surr_trca.eigenvalues[0]

    def _calculate_Q(self, data):
        # Concatenate all trials so that we can calculate covariance matrix
        # Data should be concatenated along the last axis
        continuous = data.copy()
        ntrials, nchannels, nsamples = continuous.shape
        continuous = np.concatenate(continuous, axis=-1)
        Q = (1/ntrials)*(continuous @ continuous.T)
        return Q
        
    def _apply_normalization(self, data):
        # Normalize each channel to have zero mean and unit variance
        data = (data - np.mean(data, axis=2, keepdims=True)) / np.std(data, axis=2, keepdims=True)
        return data
    
    def _calculate_U(self, data):
        data = data.copy()
        U = np.mean(data, axis=0)
        return U
    
    def _calculate_V(self, data):
        data = data.copy()
        ntrials, nchannels, nsamples = data.shape
        V = np.zeros((nchannels, nchannels))
        for trial in data:
            V += trial @ trial.T
        V = (1/ntrials)*V
        return V

    def _calculate_S(self, data, U, V):
        data = data.copy()
        ntrials, nchannels, nsamples = data.shape
        S = (1/((ntrials-1)*nsamples)) * ((U@U.T) - ((1/ntrials)*V))
        return S
    
    def _apply_inverse_regularization(self, data, reg, verbose=False):
        """ This function regularizes a given matrix and calculates it's inverse.
        Args:
            data (np.array): List of matrix to be regularized and inverted.
            reg (float): Regularization parameter.
        Returns:
            data_inverse (np.ndarray): Inverse of the Q matrix.
            S (np.ndarray): S from SVD matrix.
        """
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

# gTRCA
class gTRCA():
    """
    Group Task Related Analysis (gTRCA) class.

    Implemented according to the paper:
    Tanaka, H. (2020). Group task-related component analysis (gTRCA): A multivariate method for inter-trial reproducibility and inter-subject similarity maximization for EEG data analysis. Scientific Reports, 10(1), 84. https://doi.org/10.1038/s41598-019-56962-2

    But adapted for Evoked Potentials that are already segmented and with a 'subject' surrogate method that allows for a more robust statistical analysis for groups.

    This Python implementation was made by Couto, B.A.N. (2023)
    Member of the Neuroengineering Lab from the Federal University of São Paulo.

    For more information, see the documentation of the fit method.

    Attributes:
        times (numpy.ndarray): Time vector of the data.
        infos (list of mne.Info): List of mne.Info objects of the data.
        number_of_channels (int): Number of channels of the data.
        number_of_trials (int): Number of trials of the data.
        stimuli_onset (int): Onset of the stimuli in samples.
        evokeds (numpy.ndarray): Evoked responses of the data.
        mean_trials_covariances (numpy.ndarray): Mean trials covariance matrix of the data.
        covariance_matrices (numpy.ndarray): Covariance matrices of the data.
        s_matrix (numpy.ndarray): S matrix of the data.
        eigenvalues (numpy.ndarray): Eigenvalues of the generalized eigenvalue problem.
        eigenvectors (numpy.ndarray): Eigenvectors of the generalized eigenvalue problem.
        data (numpy.ndarray): Data used to fit the gTRCA object.
        components_with_fixed_orientation (numpy.ndarray): Indices of Components with fixed orientation.
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
        """ Fits the gTRCA object to the data. Data must be a list of mne.Epochs(...) format.

        Args:
            data (list): List of mne.Epochs(...) to fit the gTRCA object to.
            onset (float, optional) : Onset of the stimuli in seconds. Defaults to 0.
            apply_normalization (bool, optional): Whether to apply normalization to the data. Defaults to True.
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
        data = [cp.deepcopy(sub.get_data()) for sub in data]
        self.number_of_trials = [np.shape(sub)[0] for sub in data]
        self.number_of_channels = [np.shape(sub)[1] for sub in data]

        # Applying Standardization
        if verbose:
            print('Applying Standardization...')
        if standardize:
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
                      component=0,
                      subject='all',
                      average=True,
                      normalization_mode='subject',
                      normalization_window=None,
                      verbose=False):
        """ Returns projected data from subject.

        Args:
            component (int): Component to return.
            subject (int): Subject to return component from.
            average (bool): Whether to average the data across trials.
            normalization_mode (str): Normalization mode to apply. Defaults to 'subject'. Can be 'group', 'subject', or 'none'.
            normalization_window (str): Normalization window to apply. Defaults to None.
                Can be 'baseline', None or list of two integers indicating which samples to use.

        Returns:
            projections (np.ndarray): Projected data. If average is True, shape is (n_subs, n_channels, n_times). Else, returns a list with each element representing a subject with sizes (n_trial, n_channel, n_times).
            spatial_maps (np.ndarray): Spatial maps. Shape is (n_subs, n_channels)
        """

        drop_subs = lambda x: [x[i] for i in range(len(x)) if i not in self.drop_subjects]
        data = drop_subs(self.data) # Removing dropped subjects

        if subject == 'all':
            subjects = range(len(data))
        else:
            subjects = [subject]
        
        number_of_channels = drop_subs(self.number_of_channels) # Removing dropped subjects
        get_idx = lambda i: np.sum(number_of_channels[:i]) if i!=0 else 0

        projections = []
        spatial_maps = []

        if verbose:
            print(f'- Projecting Subject(s)...')
        for s in subjects:
            w = self.eigenvectors[get_idx(s):get_idx(s+1), component]
            projection = [w.T @ data[s][trial,:,:] for trial in range(np.shape(data[s])[0])]
            projection = np.array(projection)

            # Creating Spatial Map
            spatial_map = self.covariance_matrices[s] @ w

            projections.append(projection)
            spatial_maps.append(spatial_map)

        # Applying Reorientation based on 1. GMFP and 2. Average Group Component
        if verbose:
            print(f'- Checking Orientation...')
        projections, spatial_maps = self._apply_components_orientation(projections, spatial_maps, component)

        if verbose:
            print(f'- Applying Normalization...')
        projections = self._apply_components_normalization(projections,
                                                           mode=normalization_mode,
                                                           window=normalization_window)

        if average:
            projections = np.array([np.mean(proj, axis=0) for proj in projections])

        return projections, spatial_maps

    def get_average_component(self, component=0,
                               normalization_mode='subject',
                               normalization_window=None,
                               verbose=False):
        """ Calculates the Group Average Component and Spatial Map
            Args:
                component (int, optional): What component to return. Defaults to 0.
                normalization_mode (str, optional): Normalization mode to apply. Defaults to 'subject'. Can be 'group', 'subject', or 'none'.
                normalization_window (str, optional): Normalization window to apply. Defaults to None. Can be 'baseline', None or list of two integers indicating which samples to use.
                verbose (bool, optional): Whether to print out information. Defaults to False.
            Returns:
                projection (np.ndarray): Group Average Projection.
                spatial_map (np.ndarray): Group Average Spatial Map.
        """
        projections, spatial_maps = self.get_component(component,
                                                         subject='all',
                                                         average=True,
                                                         normalization_mode=normalization_mode,
                                                         normalization_window=normalization_window,
                                                         verbose=verbose
                                                         )
        projection = np.mean(projections, axis=0)
        spatial_map = np.mean(spatial_maps, axis=0)
        return projection, spatial_map

    def project(self, new_subject, component=0,
                average=True,
                reg=10**5,
                verbose=False):
        """ Projects new subject data to the gTRCA space. """""" Projects new subject data to the gTRCA space.
        
        Args:
            new_subject (mne.Epochs): New subject data to be projected.
            component (int, optional): Component to project the data to. Defaults to 0.
            verbose (bool, optional): Whether to print out information. Defaults to False.

        Returns:

        """
        # Checking if Component was Correctly Orientated
        if component not in self.components_with_fixed_orientation:
            if verbose:
                print(f'Warning: Component {component} was not oriented. Orientating now...')
            _ = self.get_projections(component=component, subject='all', average=True, verbose=verbose)

        # New subject and new subject Evoked (X_{A+1})):
        new_subject = cp.deepcopy(new_subject.get_data())
        new_subject = self._apply_standardization(new_subject)

        # New subject Covariance Matrix (Q_{A+1}):
        new_subject_cov = self._calculate_covariance_matrix(new_subject)
        new_subject_cov_inv = self._apply_inverse_regularization([new_subject_cov], reg)[0]

        # New subject filter (w_{A+1}):
        get_idx = lambda i: np.sum(self.number_of_channels[:i]) if i!=0 else 0
        get_filters = lambda i: self.eigenvectors[get_idx(i):get_idx(i+1), component]
        
        sum_of_filters_w = np.sum([(self.evokeds[i].T @ get_filters(i)) for i in range(len(self.data))], axis=0)
        new_subject_w = (1/2*len(self.data)) * (new_subject_cov_inv @ np.mean(new_subject, axis=0)) @ sum_of_filters_w

        # Projecting new subject (Y_{A+1}):
        new_subject_projection = [new_subject_w.T @ new_subject[trial, :, :] for trial in range(np.shape(new_subject)[0])]
        new_subject_projection = np.array(new_subject_projection)
        new_subject_spatial_map = new_subject_cov @ new_subject_w

        # Applying Normalization
        new_subject_average = np.mean(new_subject_projection, axis=0)
        for i in range(len(new_subject_projection)):
            new_subject_projection[i] = (new_subject_projection[i]-np.mean(new_subject_projection[i]))/np.std(new_subject_average)

        # Applying Reorientation
        correlation = np.corrcoef(np.mean(new_subject_projection, axis=0), self.get_average_projection(component)[0])[0,1]
        if correlation < 0:
            new_subject_projection *= -1
            new_subject_spatial_map *= -1
            correlation *= -1
        
        # Averaging if necessary
        if average:
            new_subject_projection = np.mean(new_subject_projection, axis=0)

        return new_subject_projection, new_subject_spatial_map, correlation

    def reorientate_to_ref(self, component, ref, verbose=True):
        """ Reorientates the components to a reference. """

        # Checking if Component was Correctly Orientated
        if component not in self.components_with_fixed_orientation:
            if verbose:
                print(f'Warning: Component {component} was not oriented. Orientating now...')
            _ = self.get_projections(component=component, subject='all', average=True, normalization='none')

        # Reorientating
        if np.corrcoef(ref, self.get_average_projection(component=component)[0])[0,1] < 0:
            self.eigenvectors[:, component] *= -1
            if verbose:
                print(f'- Component {component} Reoriented.')
        else:
            if verbose:
                print(f'- Component {component} Already Correctly Oriented.')    
        return

    def run_surrogate(self, mode='trial', minjitter=0, maxjitter='nsamples'):
        """ Runs a surrogate analysis to get the higher eigenvalue obtained with gTRCA.
        
        Args:
            mode (str, optional): Surrogate mode to use. Defaults to 'trial'. Can be 'trial' or 'subject'.
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
        self._unshift_data()
        self._calculate_matrices(reg=self.reg)
        self._apply_gtrca(reg=self.reg)
        print('Done resetting changes! ✅')

    def get_correlations(self, component=0,
                         window=[None, None]):
        """ Returns the time and spatial correlations for a given component and window, taking subjects pairwise.
        
        Args:
            component (int, optional): Component to get the correlation for. Defaults to 0.
            window (list, optional): Window to get the correlation for, in samples. Defaults to [None, None].

        Returns:
            corrs (np.array): Correlations between subjects projections.
            maps_corrs (np.array): Correlations between subjects spatial maps.
        """
        y, maps = self.get_component(component)
        corrs = np.corrcoef(y[:, window[0]:window[1]])
        maps_corrs = np.corrcoef(maps)
        idxs = np.triu_indices(np.shape(corrs)[0], k=1)
        return corrs[idxs], maps_corrs[idxs]

    def drop(self, subject=None, verbose=True):
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
        self._calculate_matrices(reg=self.reg, verbose=verbose)
        self._apply_gtrca(reg=self.reg, verbose=verbose)
        if verbose:
            print('Done running gTRCA! ✅')
            print('Subjects dropped: ', subject)
            print('Map available at .get_drop_map()')
        pass

    def get_drop_map(self):
        # Returns the map of the dropped subjects
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
        """ Calculates the S Matrix of the data.

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

    def _apply_components_orientation(self,
                                        projections,
                                        spatial_maps,
                                        component,
                                        verbose=False):
        """ This function reorientates the components and spatial maps to have ensure they are positively correlated.

        Args:
            projections (np.ndarray): Projections to be reorientated.
            spatial_maps (np.ndarray): Spatial maps to be reorientated.
            component (int): Component to reorientate. Alterates saved eigenvectors.
            verbose (bool, optional): Whether to print out information. Defaults to False.

        Returns:
            projections (np.ndarray): Reorientated projections.
            spatial_maps (np.ndarray): Reorientated spatial maps.
        """
        # Trial by Trial
        # First Direction: First Peak of GMFP
        drop_subs = lambda x: [x[i] for i in range(len(x)) if i not in self.drop_subjects]
        number_of_channels = drop_subs(self.number_of_channels) # Removing dropped subjects

        get_idx = lambda i: np.sum(number_of_channels[:i]) if i!=0 else 0
        get_evoked = lambda a: np.mean(a, axis=0)
        evokeds = [get_evoked(sub) for sub in projections]

        # Calculate the global mean field power
        gmfp = np.mean([abs(evk) for evk in evokeds], axis=0)
        gmfp_peak, _ = scipy.signal.find_peaks(gmfp, distance=len(gmfp))
        for i, evk in enumerate(evokeds):
            if evk[gmfp_peak] < 0:
                projections[i] *= -1
                spatial_maps[i] *= -1
                self.eigenvectors[get_idx(i):get_idx(i+1), component] *= -1
    
        # Second Correction: Correlation with Group Average
        evokeds = [get_evoked(sub) for sub in projections]
        mean_component = np.mean(evokeds, axis=0)
        for i, evk in enumerate(evokeds):
            if np.corrcoef(evk, mean_component)[0,1]<0:
                projections[i] *= -1
                spatial_maps[i] *= -1
                self.eigenvectors[get_idx(i):get_idx(i+1), component] *= -1

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
                    group_average = group_average[:self.stimuli_onset]
                elif window is None:
                    pass
                else:
                    group_average = group_average[window[0]:window[1]]

                # Normalization
                group_std = np.std(group_average)
                for i, projection in enumerate(data):
                    for j, trial in enumerate(projection):
                        data[i][j,:] = (trial - np.mean(trial)) / group_std

            elif mode == 'subject':
                for i, projection in enumerate(data):
                    # Subject Average
                    subject_average = get_evoked(projection)

                    # Subject Standard Deviation on chosen window
                    if window == 'baseline':
                        subject_average = subject_average[:self.stimuli_onset]
                    elif window is None:
                        pass
                    else:
                        subject_average = subject_average[window[0]:window[1]]
                    subject_std = np.std(subject_average)
                    
                    # Normalization
                    for j, trial in enumerate(projection):
                        data[i][j,:] = (trial - np.mean(trial)) / subject_std

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
