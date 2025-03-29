import mne
from pathlib import Path
from gtrca_refactor import gTRCA

files = Path("data").rglob("*epo.fif")
epochs = [mne.read_epochs(f, verbose=False) for f in files]

# gTRCA
gtrca = gTRCA()
gtrca.fit(epochs, verbose=True)

projection = gtrca.get_projections() # projection.shape = [(n_trials, n_times), ...]
maps = gtrca.get_maps() # maps.shape = (n_subs, n_channels)
