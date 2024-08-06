import fsp.methods.distance.scipy_single_thread as dist_scipy_st
import fsp.methods.distance.sklearn_multi_thread as dist_sklearn_mt
import fsp.methods.distance.torch_multi_thread_cpu as dist_torch_mt_cpu
import fsp.methods.distance.torch_gpu as dist_torch_gpu
import fsp.methods.distance.rapidsai_gpu as dist_rapidsai_gpu
import fsp.methods.kmeans.sklearn_multi_thread as kmeans_sklearn_mt
import fsp.methods.kmeans.scipy_single_thread as kmeans_scipy_st
import fsp.methods.kmeans.rapidsai_gpu as kmeans_rapidsai_gpu
from dataclasses import dataclass, asdict
from itertools import product
import json


@dataclass(frozen=True)
class Options:
    Standardize: bool = False
    initial_k: int = 1
    p_parameter: float = 0.01
    h_threshold: int = 1
    dm_case: int = 2
    s_parameter: float = 0.1
    dm_threshold: int = 3
    update_s_parameter: bool = True
    return_full_dm: bool = False
    return_full_history: bool = False
    iteration_threshold: int = 1e6
    kmeans_random_state: None | int = None
    distance_method: int = 1
    kmeans_method: int = 1

    @staticmethod
    def list1():
        # Define parameter values
        Standardize = [False, True]
        initial_k = [1, 5, 10]
        p_parameter = [0.01]
        h_threshold = [1, 10]
        dm_case = [1, 2]
        s_parameter = [0.1]
        dm_threshold = [3, 5]
        update_s_parameter = [False, True]
        return_full_dm = [False]
        return_full_history = [False]
        iteration_threshold = [int(1e6)]
        kmeans_random_state = [None]

        # Generate all combinations of the parameters
        combinations = product(
            Standardize, initial_k, p_parameter, h_threshold, dm_case,
            s_parameter, dm_threshold, update_s_parameter,
            return_full_dm, return_full_history, iteration_threshold, kmeans_random_state
        )

        # Convert combinations to a list of Options instances
        options_list = [
            Options(
                Standardize=combo[0],
                initial_k=combo[1],
                p_parameter=combo[2],
                h_threshold=combo[3],
                dm_case=combo[4],
                s_parameter=combo[5],
                dm_threshold=combo[6],
                update_s_parameter=combo[7],
                return_full_dm=combo[8],
                return_full_history=combo[9],
                iteration_threshold=combo[10],
                kmeans_random_state=combo[11]
            ) for combo in combinations
        ]

        return options_list

    def getDistanceMethod(self):
        if self.distance_method == 1:
            return dist_scipy_st
        elif self.distance_method == 2:
            return dist_sklearn_mt
        elif self.distance_method == 3:
            return dist_torch_mt_cpu
        elif self.distance_method == 4:
            return dist_torch_gpu
        elif self.distance_method == 5:
            return dist_rapidsai_gpu

    def getKMeansMethod(self):
        if self.kmeans_method == 1:
            return kmeans_sklearn_mt
        elif self.kmeans_method == 2:
            return kmeans_scipy_st
        elif self.kmeans_method == 3:
            return kmeans_rapidsai_gpu