# [arXiv:2502.20240] Entanglement buffering with multiple quantum memories

This repository contains all the code used in our manuscript 'Entanglement buffering with multiple quantum memories' ([arXiv:2502.20240](https://arxiv.org/abs/2502.20240)).

We have three main types of files:

 - `RESULTS-*.ipynb`: each of these files contains the code used to generate some of the figures shown in our paper. All plots in the paper were generated using one of these files.

 - `VALIDATION-*.ipynb`: in these files we compare the analytical calculations to simulation data (the simulation is run within the notebook files).

 - `main_1GnB.py`: this file contains all the main code, namely, mathematical description of the purification policies, simulation code, and auxiliary functions to compute and plot results.


In the folder `_bilocal_cliffords/*`, we include the code and data necessaries to compute the purification coefficients of the optimal bilocal Clifford policy (see Section IV.B and Appendix C.2 from our paper). The files in this directory require a SageMath 9.0 (or 10.0) kernel (to run these files, use `sage -n jupyter` instead of `jupyter notebook`).

 - `get_all_statistics.ipynb`: (it requires running `Jansen2022_Transversal.ipynb` first) finds the Bell-diagonal elements of all possible output states from an m-to-1 bilocal Clifford purification protocol, assuming the inputs are m Werner states. Code adapted from [1] (for the original version see https://doi.org/10.4121/15082515.v1).

 - `Jansen2022_get_all_circuits_Werner.ipynb`: (it requires running `Jansen2022_Transversal.ipynb` first) finds all combinations of output fidelity and success probability achievable by bilocal Clifford protocols. Code adapted from [1] (for the original version see https://doi.org/10.4121/15082515.v1).

 - `Jansen2022_Transversal.ipynb`: generates an auxiliary data file needed to run `get_all_statistics.ipynb`. Code adapted from [1] (for the original version see https://doi.org/10.4121/15082515.v1).

