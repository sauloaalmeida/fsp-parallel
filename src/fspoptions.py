from dataclasses import dataclass

@dataclass(frozen=True)
class FspOptions:

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

    def __str__(self):
        return f"{{initial_k={self.initial_k}, "\
               f"p_parameter={self.p_parameter}, "\
               f"h_threshold={self.h_threshold}, "\
               f"dm_case={self.dm_case}, "\
               f"dm_threshold={self.dm_threshold}, "\
               f"update_dm_threshold={self.update_dm_threshold}, "\
               f"return_full_dm={self.return_full_dm}, "\
               f"return_full_history={self.return_full_history}, "\
               f"iteration_threshold={self.iteration_threshold}, "\
               f"kmeans_random_state={self.kmeans_random_state}}}"