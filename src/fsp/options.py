import fsp.methods.distance.scipy_single_thread as dist_scipy_st
import fsp.methods.distance.sklearn_multi_thread as dist_sklearn_mt
import fsp.methods.distance.torch_multi_thread_cpu as dist_torch_mt_cpu
import fsp.methods.distance.torch_gpu as dist_torch_gpu
import fsp.methods.distance.rapidsai_gpu as dist_rapidsai_gpu
import fsp.methods.kmeans.sklearn_multi_thread as kmeans_sklearn_mt
import fsp.methods.kmeans.scipy_single_thread as kmeans_scipy_st
import fsp.methods.kmeans.rapidsai_gpu as kmeans_rapidsai_gpu
from dataclasses import dataclass, asdict
import json


@dataclass(frozen=True)
class Options:
    Standardize: bool = False
    initial_k: int = 1
    p_parameter: float = 0.05
    h_threshold: int = 1
    dm_case: int = 2
    s_parameter: float = 0.5
    dm_threshold: int = 3
    update_s_parameter: bool = True
    return_full_dm: bool = False
    return_full_history: bool = False
    iteration_threshold: float = 1e6
    kmeans_random_state: None | int = None
    distance_method: int = 1
    kmeans_method: int = 1

    @staticmethod
    def from_str(data_str):
        data = json.loads(data_str)
        return Options(**data)

    @staticmethod
    def preset(label):
        options = {
            "opt1s0": Options(p_parameter=0.01, dm_case=1, s_parameter=0.1, Standardize=0),
            "opt2s0": Options(p_parameter=0.01, dm_case=1, s_parameter=0.3, Standardize=0),
            "opt3s0": Options(p_parameter=0.01, dm_case=1, s_parameter=0.5, Standardize=0),

            "opt4s0": Options(p_parameter=0.05, dm_case=1, s_parameter=0.1, Standardize=0),
            "opt5s0": Options(p_parameter=0.05, dm_case=1, s_parameter=0.3, Standardize=0),
            "opt6s0": Options(p_parameter=0.05, dm_case=1, s_parameter=0.5, Standardize=0),

            "opt7s0": Options(p_parameter=0.01, dm_case=2, s_parameter=0.1, Standardize=0),
            "opt8s0": Options(p_parameter=0.01, dm_case=2, s_parameter=0.3, Standardize=0),
            "opt9s0": Options(p_parameter=0.01, dm_case=2, s_parameter=0.5, Standardize=0),

            "opt10s0": Options(p_parameter=0.05, dm_case=2, s_parameter=0.1, Standardize=0),
            "opt11s0": Options(p_parameter=0.05, dm_case=2, s_parameter=0.3, Standardize=0),
            "opt12s0": Options(p_parameter=0.05, dm_case=2, s_parameter=0.5, Standardize=0),

            "opt1s1": Options(p_parameter=0.01, dm_case=1, s_parameter=0.1, Standardize=1),
            "opt2s1": Options(p_parameter=0.01, dm_case=1, s_parameter=0.3, Standardize=1),
            "opt3s1": Options(p_parameter=0.01, dm_case=1, s_parameter=0.5, Standardize=1),

            "opt4s1": Options(p_parameter=0.05, dm_case=1, s_parameter=0.1, Standardize=1),
            "opt5s1": Options(p_parameter=0.05, dm_case=1, s_parameter=0.3, Standardize=1),
            "opt6s1": Options(p_parameter=0.05, dm_case=1, s_parameter=0.5, Standardize=0),

            "opt7s1": Options(p_parameter=0.01, dm_case=2, s_parameter=0.1, Standardize=1),
            "opt8s1": Options(p_parameter=0.01, dm_case=2, s_parameter=0.3, Standardize=1),
            "opt9s1": Options(p_parameter=0.01, dm_case=2, s_parameter=0.5, Standardize=1),

            "opt10s1": Options(p_parameter=0.05, dm_case=2, s_parameter=0.1, Standardize=1),
            "opt11s1": Options(p_parameter=0.05, dm_case=2, s_parameter=0.3, Standardize=1),
            "opt12s1": Options(p_parameter=0.05, dm_case=2, s_parameter=0.5, Standardize=1)
        }
        if label not in options:
            raise ValueError('Option informed not recognized.')

        return options[label]

    def __str__(self):
        return json.dumps(asdict(self))

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