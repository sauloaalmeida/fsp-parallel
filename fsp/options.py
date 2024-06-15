from dataclasses import dataclass

@dataclass(frozen=True)
class Options:

    initial_k: int = 1
    p_parameter: float = 0.05
    h_threshold: float = 0
    dm_case: int = 2
    dm_threshold: float = 0.5
    update_dm_threshold: bool = True
    return_full_dm: bool = False
    return_full_history: bool = False
    iteration_threshold: float = 2e6
    kmeans_random_state: int | None = None