from typing import Tuple, Union

import numpy as np

PATH_TO_MATRIX_GAME_DECOMPOSITION = "/home/oberlechner/projects/game_decomposition"


def determine_alpha(potentialness: float, uP: np.ndarray, uH: np.ndarray) -> float:
    """Instead of computing the potentialness for a given alpha.
    we want to determine the alpha for a given potentialness

    Args:
        potentialness (float): value of potentialness metric
        uP (np.ndarray): potential part
        uH (np.ndarray): harmonic part

    Returns:
        float: value for alpha, i.e., convex combination of parts
    """
    if potentialness is None:
        return 1 / 2
    else:
        assert 0 <= potentialness <= 1
        return (
            potentialness
            * np.linalg.norm(uH)
            / (
                (1 - potentialness) * np.linalg.norm(uP)
                + potentialness * np.linalg.norm(uH)
            )
        )


def get_payoff_matrix(
    game: str,
    n_actions: int,
    potentialness: float = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Construct game out of convex combination of potential and harmonic part

    Args:
        game (str): name of game
        n_actions (int): discretization
        potentialness (float): desired potentialness for game

    Returns:
        Tuple[np.ndarray, np.ndarray, float]: payoff matrices for both agents
    """

    uP = load_component(game, n_actions, "uP")
    uH = load_component(game, n_actions, "uH")
    alpha = determine_alpha(potentialness, uP, uH)

    payoff = {}
    for agent in [1, 2]:
        uH_agent = vector_to_matrix(uH, agent=agent, n_agents=2, n_actions=n_actions)
        uP_agent = vector_to_matrix(uP, agent=agent, n_agents=2, n_actions=n_actions)
        payoff[agent] = alpha * uP_agent + (1 - alpha) * uH_agent
    return payoff[1], payoff[2], alpha


def load_component(game: str, n_actions: int, component: str) -> np.ndarray:
    """load computed component"""

    if component not in ["payoff", "uH", "uN", "uP"]:
        raise ValueError(
            f"component {component} not available (choose from payoff, uH, uP, uN)"
        )
    try:
        file = f"{PATH_TO_MATRIX_GAME_DECOMPOSITION}/results/{game}/discr_{n_actions}/components/{component}.npy"
        return np.load(file)
    except:
        print(
            f"component {component} of game {game} with {n_actions} actions not found"
        )
        print(
            f"{PATH_TO_MATRIX_GAME_DECOMPOSITION}/results/{game}/discr_{n_actions}/components/{component}.npy"
        )
        return None


def vector_to_matrix(
    u: Union[list, np.ndarray], agent: int, n_agents: int, n_actions: int
) -> np.ndarray:
    """get matrix (multi-dim for >2 agents) from vector

    Args:
        u (Union[list, np.ndarray]): vector ,e.g. payoff
        agent (int): agent (starts at 1)
        n_agents (int): total number of agents
        n_actions (int): number of discrete ections

    Returns:
        np.ndarray: _description_
    """
    if isinstance(u, list):
        u = np.array(u).astype(float)
    assert len(u) == n_agents * n_actions**n_agents

    matrix_entries = u[
        (agent - 1) * n_actions**n_agents : agent * n_actions**n_agents
    ]
    return matrix_entries.reshape(tuple([n_actions] * n_agents))


def potentialness_game(game_name: str, n_actions: int):
    uP = load_component(game_name, n_actions, "uH")
    uH = load_component(game_name, n_actions, "uH")
    return metric_potentialness(uP, uH)


def metric_potentialness(uP: np.ndarray, uH: np.ndarray) -> float:
    """Compute potentialness of game

    Args:
        uP (np.ndarray): potential component
        uH (np.ndarray): harmonic component

    Returns:
        float: |uP| / (|uP| + |uH|)
    """
    if (uP is None) or (uH is None):
        return np.nan
    else:
        return np.linalg.norm(uP) / (np.linalg.norm(uP) + np.linalg.norm(uH))
