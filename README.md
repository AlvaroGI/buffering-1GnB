# Entanglement buffering with multiple quantum memories

This repository contains all the code used in our manuscript 'Entanglement buffering with multiple quantum memories' (see the arxiv version of our manuscript at ...).

We have three types of file:

 - `

`_bilocal_cliffords/*`: to run these files, use `sage -n jupyter` instead of `jupyter notebook`.
 - `Jansen2022_Transversal.ipynb`: generates an auxiliary data file needed to run `get_all_circuits_1G1B.ipynb`. This notebook requires a SageMath 9.0 (or 10.0) kernel. Code adapted from [1] (for the original version see https://doi.org/10.4121/15082515.v1).
 - `Jansen2022_get_all_circuits_1G1B.ipynb`: (it requires running `Jansen2022_Transversal.ipynb` first) finds all combinations of output fidelity and success probability achievable by bilocal Clifford protocols, and saves the results in a matlab-readable (.mat) file. This notebook requires a SageMath 9.0 (or 10.0) kernel. Code adapted from [1] (for the original version see https://doi.org/10.4121/15082515.v1).