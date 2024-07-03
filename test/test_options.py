import pytest
from dataclasses import FrozenInstanceError
from fsp.options import FspOptions

class TestFspOptions:

    def test_default_values(self):
        options = FspOptions()
        assert options.Standardize == False
        assert options.initial_k == 1
        assert options.p_parameter == 0.05
        assert options.h_threshold == 1
        assert options.dm_case == 2
        assert options.s_parameter == 0.5
        assert options.dm_threshold == 3
        assert options.update_s_parameter == True
        assert options.return_full_dm == False
        assert options.return_full_history == False
        assert options.iteration_threshold == 1e6
        assert options.kmeans_random_state == None

    def test_not_default_values(self):
        options = FspOptions(initial_k = 2, dm_case = 3,  kmeans_random_state = 0)
        assert options.Standardize == False
        assert options.initial_k == 2
        assert options.p_parameter == 0.05
        assert options.h_threshold == 1
        assert options.dm_case == 3
        assert options.s_parameter == 0.5
        assert options.dm_threshold == 3
        assert options.update_s_parameter == True
        assert options.return_full_dm == False
        assert options.return_full_history == False
        assert options.iteration_threshold == 1e6
        assert options.kmeans_random_state == 0

    def test_modify_attribute(self):
        with pytest.raises(FrozenInstanceError):
            options = FspOptions()
            options.dm_case = 3

    def test_add_attribute(self):
        with pytest.raises(FrozenInstanceError):
            options = FspOptions()
            options.dm_case2 = 2
