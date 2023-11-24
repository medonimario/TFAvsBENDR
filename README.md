# Modern Approaches in Representing Inter­brain Oscillations: A Comparative Study of BENDR and TFA in Two­Brain EEG Analysis

This repository contains the code described in the methodology section of my bachelor project. It was used to:
- Preprocess raw hyperscanning EEG data (The folder Pre-processing all pairs contains the individual jupyter notebooks for each of the 21 pairs with their respective ICAs being displayed for manual review)
- Obtain feature vectors from TFA representations, storing them in dataframes
- Add a 10Hz synthetic wave to the dataset and obtain a new set of time-frequency representations
- Obtain accuracy scores from both TFA and BENDR feature vectors across 4 different classifiers, both on experimental and synthetic datasets
- Perform statistical analysis on the classifier accuracy scores

This repository misses the implementation of the BENDR model to obtain feature vectors, but that can be found on my separate [BENDR fork](https://github.com/medonimario/BENDR).
