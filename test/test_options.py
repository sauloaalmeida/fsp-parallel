import pytest
import dataclasses
from dataclasses import FrozenInstanceError
from fsp.options import Options

class TestOptions:

    def test_default_values(self):
        options = Options()
        assert options.Standardize == False
        assert options.initial_k == 1
        assert options.p_parameter == 0.01
        assert options.h_threshold == 1
        assert options.dm_case == 2
        assert options.s_parameter == 0.1
        assert options.dm_threshold == 3
        assert options.update_s_parameter == True
        assert options.return_full_dm == False
        assert options.return_full_history == False
        assert options.iteration_threshold == 1e6
        assert options.kmeans_random_state == None
        assert options.distance_method == 3
        assert options.kmeans_method == 2

    def test_not_default_values(self):
        options = Options(initial_k = 2, dm_case = 3,  kmeans_random_state = 0, kmeans_method = 3, distance_method = 1)
        assert options.Standardize == False
        assert options.initial_k == 2
        assert options.p_parameter == 0.01
        assert options.h_threshold == 1
        assert options.dm_case == 3
        assert options.s_parameter == 0.1
        assert options.dm_threshold == 3
        assert options.update_s_parameter == True
        assert options.return_full_dm == False
        assert options.return_full_history == False
        assert options.iteration_threshold == 1e6
        assert options.kmeans_random_state == 0
        assert options.distance_method == 1
        assert options.kmeans_method == 3

    def test_modify_attribute(self):
        with pytest.raises(FrozenInstanceError):
            options = Options()
            options.dm_case = 3

    def test_add_attribute(self):
        with pytest.raises(FrozenInstanceError):
            options = Options()
            options.dm_case2 = 2