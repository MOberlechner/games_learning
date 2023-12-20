# Project: HODGE


### Computation
All experiments are stored in `projects/hodge/data/`.

**Experiment 1 - Random Games** <br>
Generate $n=10^6$ random games by sampling entries of the payoff matrices from $U([0,1])$ and compare potentialness for different number of actions and agents
```python
python projects/hodge/computation/run_random_potentialness.py
```
*Note: We can also sample utilties from a normal distribution $\mathcal{N}(0,1)$, but so far we couldn't observe differences in the distribution.*

From the sampled random games, we can sample a subset of games with a certain level of potentialness and analyze behavior of mirror ascent in these games.
```python
python projects/hodge/computation/run_random_learning.py
```

**Experiment 2 - Econ Games** <br>

