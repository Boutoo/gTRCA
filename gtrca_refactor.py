import mne
import numpy as np
from scipy.linalg import block_diag
from tqdm import tqdm


# gTRCA
class gTRCA():
    def __init__(self):
        self.data = None
        self.nsubs = None
        self.times = None
        self.S = None
        self.Q = None
        self.fitted = False
        self.eigenvalues = None
        self.eigenvectors = None
        self.verbose = True

    def __str__(self):
        fitted = False if self.data is None else True
        nsubs = len(self.data) if fitted else None
        return f"gTRCA {'fitted ' if fitted else 'unfitted '}{f'nsubs={nsubs}' if fitted else ''}"
    
    def __repr__(self):
        return '<'+ self.__str__()+'>'
    
    def fit(self, epochs_list, verbose=True):
        epochs_list = [ep.copy().pick('eeg') for ep in epochs_list]
        self.nsubs = len(epochs_list)
        self.nchs = [len(epochs_list[sub].ch_names) for sub in range(self.nsubs)]
        self.times = epochs_list[0].times
        
        # check if all epochs have the same number of time points
        if not all([len(epochs_list[sub].times) == len(self.times) for sub in range(self.nsubs)]):
            raise ValueError("All epochs must have the same number of time points")
        ntimes = len(self.times)

        # Normalize (z-score) along the samples axis for each trial
        data = [epochs_list[sub].get_data(copy=True) for sub in range(self.nsubs)]
        pbar = tqdm(data, desc="Normalizing data", disable=not verbose, colour="#72968E")
        self.data = [(ep - np.mean(ep, axis=(0,2), keepdims=True)) / np.std(ep, axis=(0,2), keepdims=True) for ep in pbar]
        self.Q = gTRCA.compute_q_matrix(self.data, verbose)
        self.S = gTRCA.compute_s_matrix(self.data, verbose)
        inv_q, _ = gTRCA.regularized_inverse(self.Q)
        eigenvalues, eigenvectors = np.linalg.eig(inv_q @ self.S)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        reg_num = gTRCA.regularize(abs(eigenvalues))
        eigenvalues = eigenvalues[:reg_num]
        eigenvectors = eigenvectors[:,:reg_num]
        if np.max(abs(np.imag(eigenvalues))) != 0:
            print("Warning: Complex eigenvalues found")
        self.eigenvalues = eigenvalues.real
        self.eigenvectors = eigenvectors.real
        self.fitted = True

    def get_projections(self, component:int=0, response:tuple[int|None]=(0, None), fix_polarity:bool|int=True, average:bool=False):
        ntimes = len(self.times)
        eigenvector = self.eigenvectors[:, component]
        projected = []
        for sub in range(self.nsubs):
            proj = np.zeros((self.data[sub].shape[0], ntimes))
            for trial in range(self.data[sub].shape[0]):
                proj[trial] = eigenvector[sub*self.nchs[sub]:(sub+1)*self.nchs[sub]] @ self.data[sub][trial]
            projected.append(proj)

        # Flip polarity based on the max absolute value
        if fix_polarity == True:
            projected = [proj * -1 if np.abs(np.min(np.mean(proj,0))) > np.abs(np.max(np.mean(proj, 0))) else proj for proj in projected]
        elif fix_polarity == 1:
            projected = [proj * np.sign(np.max(np.mean(proj, 0))) for proj in projected]
        elif fix_polarity == -1:
            projected = [proj * np.sign(np.min(np.mean(proj, 0))) for proj in projected]

        # Flip polarity based on the group template
        if response is not None:
            response_idx = np.logical_and(
                self.times >= (response[0] if response[0] is not None else -np.inf),
                self.times < (response[1] if response[1] is not None else np.inf)
                )
            group_template = np.mean([np.mean(proj, axis=0) for proj in projected], axis=0)[response_idx]
            for i in range(len(projected)):
                projected[i] = np.sign(np.corrcoef(projected[i].mean(0)[response_idx], group_template)[0, 1]) * projected[i]

        if average:
            return np.array([np.mean(proj, axis=0) for proj in projected])
        
        return projected

    def get_maps(self, component=0, fix_polarity=True):
        eigenvector = self.eigenvectors[:, component]
        maps = []
        for sub in range(self.nsubs):
            map_ = self.Q[
                sub*self.nchs[sub]:(sub+1)*self.nchs[sub],
                sub*self.nchs[sub]:(sub+1)*self.nchs[sub]] @ \
                    eigenvector[sub*self.nchs[sub]:(sub+1)*self.nchs[sub]]
            maps.append(map_)
        
        if fix_polarity == True:
            # Flip polarity based on the max absolute value
            maps = [map_ * -1 if np.abs(np.min(map_)) > np.abs(np.max(map_)) else map_ for map_ in maps]    
        elif fix_polarity == 1:
            maps = [map_ * np.sign(np.max(map_)) for map_ in maps]
        elif fix_polarity == -1:
            maps = [map_ * np.sign(np.min(map_)) for map_ in maps]

        group_template = np.mean(maps, axis=0)
        for i in range(len(maps)):
            maps[i] = np.sign(np.dot(maps[i], group_template)) * maps[i]

        if len(set(self.nchs)) == 1:
            maps = np.array(maps)

        return maps
    
    @staticmethod
    def compute_s_matrix(data, verbose=True):
        nsubs = len(data)
        nchs = [data[s].shape[1] for s in range(nsubs)]
        get_idx = lambda i: np.sum(nchs[:i]) if i!=0 else 0

        ntimes = data[0].shape[-1]
        pbar = tqdm(data, desc="Computing S matrix main diagonal", disable=not verbose, colour="#72968E")
        S = block_diag(*[gTRCA._compute_s_matrix(ep) for ep in pbar]) * 2
        U = [ep.mean(0) for ep in data]
        pbar = tqdm(range(len(data)), desc="Computing S matrix off diagonals", disable=not verbose, colour="#72968E")
        for a in pbar:
            for b in range(a, nsubs):
                if a == b:
                    continue
                S[get_idx(a):get_idx(a+1), get_idx(b):get_idx(b+1)] = (U[a] @ U[b].T) / ntimes
                S[get_idx(b):get_idx(b+1), get_idx(a):get_idx(a+1)] = (U[b] @ U[a].T) / ntimes
        return S

    @staticmethod
    def _compute_s_matrix(data):
        ntrials, nchs, nsamples = data.shape
        U = np.mean(data, axis=0)
        V = np.mean([np.cov(k) for k in data], axis=0)
        S = (ntrials/((ntrials-1)*nsamples)) * (U @ U.T - V/ntrials)
        return S

    @staticmethod
    def compute_q_matrix(data, verbose=True):
        nsubs = len(data)
        ntrials = [data[s].shape[0] for s in range(nsubs)]
        nchs = [data[s].shape[1] for s in range(nsubs)]
        ntimes = data[0].shape[-1]
        pbar = tqdm(enumerate(data), total=len(data), desc="Computing Q matrix", disable=not verbose, colour="#72968E")
        Q = [np.cov(np.hstack(data[s])) for s, ep in pbar]
        Q = block_diag(*Q)
        return Q

    @staticmethod
    def regularized_inverse(mat, reg=1e5):
        U, SVD, V = np.linalg.svd(mat)
        reg_num = gTRCA.regularize(SVD, reg)
        data_inverse = (V[0:reg_num, :].T* (1./SVD[0:reg_num])) @ U[:, 0:reg_num].T
        return data_inverse, SVD

    @staticmethod
    def regularize(mat, reg=1e5):
        eps = np.finfo(float).eps
        ix1 = np.where(np.abs(mat)<1000*np.finfo(float).eps)[0] # Removing null components
        ix2 = np.where((mat[:-1] / (eps+mat[1:])) > reg)[0] # Cut-off based on eingenvalue ratios
        ix = np.union1d(ix1,ix2)
        return len(mat) if len(ix)==0 else np.min(ix)

    