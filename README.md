# Learning in Matrix Games

To run the experiments we need the components of the decompositions of the games we want to consider. To access them, you have to define the path to the matrix_game_decomposition in  _src.util.matrix_game_decomposition_, e.g., 

`PATH_TO_MATRIX_GAME_DECOMPOSITION = "/home/oberlechner/projects/game_decomposition"`

In the directory _scripts_, you can find the scripts to run the experiments. They have to be started from within the directory, i.e., `cd scripts`.

1. To run the actual learning experiments run: `python run_learning.py` <br>
    You can choose between different settings and specify parameters for the experiment
    - n_actions: discretization of action space
    - number of runs: how often do we repeat each setting with different initializations
    - num_pot_level: for how many levels of potentialness do we want to run the experiments

    And you can also specify parameters of the learning algorithm
    - max_iter: number of maximal iterations for the learning method
    - tol: stopping criterion (util_loss < tol), which determines convergence
    - eta: (initial) step size for learning algorithm: eta_t = eta * 1/iter**beta

2. After running the experiments you can visualize the results via: `python visualize_learning_results.py` <br>
Note that you have to have the same parameters as definied in the run_learning.py script. 
To visualize the potentialness of the underlying original game, you also have to run `python get_potentialness.py` which creates a csv in _results/data_ with the respective entries.

## Setup

Note: These setup instructions assume a Linux-based OS and uses python 3.11 (or higher).

Install virtualenv (or whatever you prefer for virtual envs)

`sudo apt-get install virtualenv`

Create a virtual environment with virtual env (you can also choose your own name)

`virtualenv -p python3.11 venv`

You can specify the python version for the virtual environment via the -p flag. 
Note that this version already needs to be installed on the system (e.g. `virtualenv - p python3.11 venv` uses the 
standard python3 version from the system).

activate the environment with

`source ./venv/bin/activate`

Install all requirements

`pip install -r requirements.txt`

## Install pre-commit hooks (for development)
Install pre-commit hooks for your project

`pre-commit install`

Verify by running on all files:

`pre-commit run --all-files`

For more information see https://pre-commit.com/.
