# gTRCA for Python

The gTRCA for Python project is an implementation of the Group Task-Related Component Analysis (gTRCA) method [(Tanaka, 2020)]() for Python. This method is a multivariate analysis technique designed to identify the neural components that are consistently task-related across a group of participants. It extends the conventional TRCA by incorporating between-subjects variability, making it suitable for group-level analysis. The gTRCA method has applied to TMS-EEG data on a study to investigate the [group level reproducibility of TMS evoked potentials (Couto et al., 2023)]() and this project is specifically tailored for such purposes. The implementation is designed to work with MNE.Epochs() objects, a commonly used data structure for handling EEG data in Python.

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

# Running gTRCA
gtrca = gTRCA(list_of_epochs, ncomps=1)
gtrca.plot_ydata()

# Creating a single surrogate
surrogate = create_surrogate(list_of_epochs, mode='trial')
```

## License
This project is licensed under the MIT License.

## References
* Tanaka, 2020
* Couto, 2023