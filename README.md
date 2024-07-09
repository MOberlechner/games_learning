# Learning in Normal-Form Games
This repo focuses on matrix games and learning algorithms.

### What is implemented?
The implementation focuses on 
- **Matrix Games**
    - Randomly generated Matrix Games
    - Economic Games (complete-information auctions, contests, and oligopolies)

- **Learning Algorithms**
    - Online Mirror Descent 
    - tbd

### Projects

|  | Description |
| ------- | ----------- |
| [**tbd**](./projects/)<br>| ... |



## Setup
<details><summary>
Note: These setup instructions assume a Linux-based OS and uses python 3.11 (or higher).
</summary>
Install virtualenv (or whatever you prefer for virtual envs)
```
sudo apt-get install virtualenv
```
Create a virtual environment with virtual env (you can also choose your own name)
```
virtualenv -p python3 venv
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


**Install pre-commit hooks (for development)**<br>
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
