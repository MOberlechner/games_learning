from games_learning.game.econ_game import (
    Bertrand,
    BertrandLinear,
    BertrandLogit,
    BertrandStandard,
)


def get_game(demand: str, n_discr: int, n_agents: int = 2) -> Bertrand:
    """Create Bertrand Game"""

    if demand == "standard":
        return BertrandStandard(
            n_agents=n_agents,
            n_discr=n_discr,
            cost=0.0,
            interval=(0.1, 1.0),
            maximum_demand=1.0,
        )

    elif demand == "linear":
        return BertrandLinear(
            n_agents=n_agents,
            n_discr=n_discr,
            cost=0.0,
            interval=(0.1, 1.0),
            alpha=0.48,
            beta=0.9,
            gamma=0.6,
        )

    elif demand == "logit":
        return BertrandLogit(
            n_agents=n_agents,
            n_discr=n_discr,
            cost=1.0,
            interval=(1.0, 2.5),
            alpha=2.0,
            mu=0.25,
        )

    else:
        raise ValueError(
            f"demand model {demand} unknown. Choose from: standard, linear, logit"
        )
