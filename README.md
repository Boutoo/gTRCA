# gTRCA for Evoked Potentials

The gTRCA for Python project is an implementation of the Group Task-Related Component Analysis (gTRCA) method [(Tanaka, 2020)](https://www.nature.com/articles/s41598-019-56962-2) for Evoked Potentials using Python. This method is a multivariate analysis technique designed to identify the neural components that are consistently task-related across a group of participants. It extends the conventional TRCA by incorporating between-subjects variability, making it suitable for group-level analysis. This version of the gTRCA method has been applied to Auditory Evoked Potentials as well as to TMS-EEG data to investigate the [group level reproducibility of TMS evoked potentials (Under development) ](https://github.com/Boutoo/gTRCA). The current implementation is designed to work with a list of MNE.Epochs() objects, a commonly used data structure for handling EEG data in Python.

## Features

- A `gTRCA()` class that accepts a list of MNE.Epochs() objects and returns an object containing a specified number of components.
- A function to create surrogate data (trial or subject-based shifting).


## Dependencies

- mne
- scipy
- numpy
- matplotlib
- copy

## Installation

To install the required libraries, you can use either `conda` or `pip`.

```bash
# Using conda
conda install mne scipy numpy matplotlib

# Using pip
pip install mne scipy numpy matplotlib
```

## Usage
Here's an example of how to use gTRCA for Python:

```python
from gtrca import gTRCA
import mne

#list_of_epochs = [mne.Epochs(...), ...]

# Running gTRCA
gtrca = gTRCA()
gtrca.fit(list_of_epochs)

# Creating a single surrogate
surrogate = gtrca.run_surrogate(mode='subject')
```

## License
This project is licensed under the MIT License.

## References
* Tanaka, H. Group task-related component analysis (gTRCA): a multivariate method for inter-trial reproducibility and inter-subject similarity maximization for EEG data analysis. Sci Rep 10, 84 (2020). https://doi.org/10.1038/s41598-019-56962-2
* Under Development: Couto, B.A.N, et al. Extracting reproducible components from TMS-EEG recordings with Group Task-related Component Analysis. Coming up on 2024.
* Couto, B.A.N. and Casali, A. G., Classification of Auditory Oddball Evoked Potentials using Group Task Related Component Analysis. Submitted to the 10th International Conference on Biomedical Engineering and Systems (ICBES 2023)
