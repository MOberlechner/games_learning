# Project: HODGE <br>

> **Characterizing the Convergence of Game Dynamics via Potentialness** <br> 
>*M. Bichler, D. Legacci, P. Mertikopoulos, M. Oberlechner, B. Pradelski*, 2025

This project focuses on the connection between learnable equilibria and the potentialness of the game.
Additional to the packages in [requirements.txt](../../requirements.txt), we also need the [decomposition](https://github.com/MOberlechner/games_decomposition) package which computes the hodge decomposition of a matrix game.
Clone the project and install the package via
```python
pip install -r requirements.txt
pip install -e ../games_decomposition
```
---
**To reproduce all experiments and figures from the submission run the following scripts.**
- start the scripts from the directory of the repository, i.e., `path_user/matrix_game_learning`.
- Parameters for experiments and visualizations can be found in `projects/hodge/configs.py`. <br>
- All experiments are stored in `projects/hodge/data/`. <br>

## Matrix Games
**Computation**
- generate data for Table 1 (_data/matrix_games/potentialness.csv_)
```python
python projects/hodge/computation/run_matrix_games_potentialness.py 
```
**Evaluation**
- generate Figure 1
```python
projects/hodge/evaluation/plot_decompositions.py 
```

## Runtime
**Computation**
- generate data with runtime for computation of potentialness and necessary operators (_data/runtime/flow_space.csv_)
```python
python projects/hodge/computation/run_runtime.py 
```
**Evaluation**
- generate Figure 2
```python
projects/hodge/evaluation/plot_runtime.py 
```

## Random Games
**Computation**
- generate data with potentialness and pure equilibria of 10^6 random games (_data/random_flow_1e6/*_)
- generate data on convergence of OMD for some randomly generated games with fixed intial strategies (_data/random_learning_1run/*_) and randomly sampled initial strategies (_data/random_learning_25run/*)
```python
python projects/hodge/computation/run_random_games_potentialness.py 
python projects/hodge/computation/run_random_games_learning.py 
```
**Evaluation**
- generate Figure 3, 4 and 
- generate Figure 6 and 9
```python
python projects/hodge/evaluation/plot_random_games_potentialness.py 
python projects/hodge/evaluation/plot_random_games_learning.py 
```
## Economic Games
**Computation**
- generate data with potentialness for different economic games with different discretizations and valuations (_data/econgames/potentialness.csv_)
- generate data with convergence of OMD on games created by different convex combinations of the harmonic and potential part by economic games (_data/econgames/learning_2_11_20bins_100runs.csv_)
```python
python projects/hodge/computation/run_econ_games_potentialness.py 
python projects/hodge/computation/run_econ_games_learning.py 
```
**Evaluation**
- create Figure 7
- create Figure 8
```python
python projects/hodge/evaluation/plot_econ_games_potentialness.py 
python projects/hodge/evaluation/plot_econ_games_learning.py 
```

