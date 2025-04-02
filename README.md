# gTRCA for Evoked Potentials

The **gTRCA for Python** project is an implementation of the Group Task-Related Component Analysis (gTRCA) method [(Tanaka, 2020)](https://www.nature.com/articles/s41598-019-56962-2) for analyzing evoked potentials using Python. gTRCA is a multivariate analysis technique designed to identify neural components that are consistently task-related across a group of participants. It extends conventional TRCA by incorporating between-subject variability, making it suitable for group-level analysis.

This version of the gTRCA method has been applied to both **Auditory Evoked Potentials** and **TMS-EEG data** to investigate the group-level reproducibility of TMS-evoked potentials. The current implementation is designed to work with a list of `mne.Epochs` objects, a commonly used data structure for handling EEG data in Python.

Please refer to:  
**Couto et al. (2025)** – *Extracting reproducible components from TMS-EEG recordings with Group Task-Related Component Analysis* (in preparation)

---

## Features

- A `gTRCA()` class that accepts a list of `mne.Epochs` objects and returns a specified number of group-level reproducible components.
- Several methods to manipulate gTRCA components, including the generation of surrogate datasets (via trial- or subject-based circular shifting). See *Couto et al. (2025)* for details.

---

## Dependencies

- `mne`
- `scipy`
- `numpy`
- `matplotlib`

---

## Installation

You can install the required libraries using either `conda` or `pip`:

```bash
# Using conda
conda install mne scipy numpy matplotlib

# Using pip
pip install mne scipy numpy matplotlib
```

## Usage
Here's an example of how to use gTRCA for Python:

```Python
from gtrca import gTRCA
import mne

# list_of_epochs = [mne.Epochs(...), ...]

# Running gTRCA
gtrca = gTRCA()
gtrca.fit(list_of_epochs)

# Extracting projections (time-courses) and spatial maps of the first component
projection, spatial_map = gtrca.get_component(component=[0])

# Creating a single surrogate dataset (subject-level shifting)
surrogate = gtrca.run_surrogate(mode='subject')
```

## License
This project is licensed under the MIT License.

## References
* Tanaka, H. Group task-related component analysis (gTRCA): a multivariate method for inter-trial reproducibility and inter-subject similarity maximization for EEG data analysis. Sci Rep 10, 84 (2020). https://doi.org/10.1038/s41598-019-56962-2
* Couto, B.A.N, et al.  Extracting reproducible components from TMS-EEG recordings with Group Task-related Component Analysis.(2025 - in preparation)
* Couto, B.A.N. and Casali, A. G., Classification of Auditory Oddball Evoked Potentials using Group Task Related Component Analysis. 10th International Conference on Biomedical Engineering and Systems (ICBES 2023)
