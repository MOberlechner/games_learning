# Learning in Normal-Form Games
This repo focuses on matrix games and learning algorithms.
Specifically, it provides a few tools to analyze the matrix games (e.g., compute pure Nash equilibria, find dominated actions) and to run gradient-based learning algorithms (e.g. mirror descent).

### What is implemented?
The implementation focuses on 
- **Matrix Games**
    - Randomly generated Matrix Games
    - Economic Games (complete-information auctions, contests, and oligopolies)
    - Bayesian Games (transforms discrete Bayesian games in matrix games)

- **Learning Algorithms**
    - Online Mirror Descent (with entropic regularizer)
    - Best Response Dynamics

### Projects

| Project | Description |
| ------- | ------------------------------------------ |
| [**hodge**](./projects/hodge/)<br>| **Characterizing the Convergence of Game Dynamics via Potentialness** <br> *M. Bichler, D. Legacci, P. Mertikopoulos, M. Oberlechner, B. Pradelski*, 2025  <br> We analyze the connection of potentialness and convergence of learning algorithms in matrix games. We focus on randomly generated games and complete-information economic games (e.g., auctions, contests). |

## Setup
<details><summary>
Note: These setup instructions assume a Linux-based OS and uses python 3.11 (or higher).
</summary>
Install virtualenv (or whatever you prefer for virtual environments)

```
sudo apt-get install virtualenv
```
Create a virtual environment with virtual env (you can also choose your own name)

```
virtualenv -p python3 venv_games
```
You can specify the python version for the virtual environment via the -p flag. 
Note that this version already needs to be installed on the system (e.g. `virtualenv - p python3 venv` uses the 
standard python3 version from the system).

activate the environment with
```
source ./venv/bin/activate
```

Install all requirements

```
pip install -r requirements.txt
```

Install the decomposition package.

```
pip install -e .
```

You can also run "pip install ." if you don't want to edit the code. The "-e" flag ensures that pip does not copy the code but uses the editable files instead.


**For Development, install pre-commit hooks**<br>
Install pre-commit hooks for your project

```
pre-commit install
```
Verify by running on all files:
```
pre-commit run --all-files
```

For more information see https://pre-commit.com/.
</details>
