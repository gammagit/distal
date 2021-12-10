This repository contains code as well as data related to:

Gaurav Malhotra, Marin Dujmovic, John Hummel & Jeffrey Bowers. The contrasting shape representations that support objectrecognition in humans and CNNs

Code related to generating datasets as well as simulating CNNs is contained in the `simulations` directory.

Experiment data is contained in the `experiment` directory.

Description of key files:
- `./simulations/create_stim_exp1.py`: Script used to create the stimuli set for the Hummel & Stankiewicz (1996) experiments.
- `./simulations/create_stim_exp2.py`: Script used to create the stimuli set for the Polygon experiment
- `./simulations/sim_exp1.py`: Script used to train and test CNNs on Hummel & Stankiewicz (1996) stimuli
- `./simulations/sim_exp2.py`: Script used to train and test CNNs on single part objects (Experiment 2)
- `./simulations/compute_similarity.ipynb`: Python notebook to generate cosine similarity between internal representations for Basis and two deformations.
- `./experiment/exp2_data.csv`: Data for all participants (anonymised). Each row corresponds to a trial. The conditions are labelled 'base', 'shear1', 'shear2' for Basis, D1 and D2 deformations that make relational change and 'rot1', 'rot2' for D1 and D2 deformations that make coordinate change.

All datasets used to run the simulations listed in the manuscript are included in the directory `./simulations/data`