import pytest
from dataclasses import FrozenInstanceError
from src.fspoptions import FspOptions


class TestFspOptions:

    def test_default_values(self):
        fspOptions = FspOptions()
        assert fspOptions.initial_k == 1
        assert fspOptions.p_parameter == 0.05
        assert fspOptions.h_threshold == 0
        assert fspOptions.dm_case == 2
        assert fspOptions.dm_threshold == 0.5
        assert fspOptions.update_dm_threshold == True
        assert fspOptions.return_full_dm == False
        assert fspOptions.return_full_history == False
        assert fspOptions.iteration_threshold == 2e6
        assert fspOptions.kmeans_random_state == None

    def test_not_default_values(self):
        fspOptions = FspOptions(initial_k = 2, dm_case = 3,  kmeans_random_state = 0)
        assert fspOptions.initial_k == 2
        assert fspOptions.p_parameter == 0.05
        assert fspOptions.h_threshold == 0
        assert fspOptions.dm_case == 3
        assert fspOptions.dm_threshold == 0.5
        assert fspOptions.update_dm_threshold == True
        assert fspOptions.return_full_dm == False
        assert fspOptions.return_full_history == False
        assert fspOptions.iteration_threshold == 2e6
        assert fspOptions.kmeans_random_state == 0

    def test_modify_attribute(self):
        with pytest.raises(FrozenInstanceError):
            fspOptions = FspOptions()
            fspOptions.dm_case = 3

    def test_add_attribute(self):
        with pytest.raises(FrozenInstanceError):
            fspOptions = FspOptions()
            fspOptions.dm_case2 = 2
