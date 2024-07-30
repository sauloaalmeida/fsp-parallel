import pytest
import dataclasses
from dataclasses import FrozenInstanceError
from fsp.options import Options

class TestFspOptions:

    def test_default_values(self):
        options = Options()
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
        assert options.distance_method == 1
        assert options.kmeans_method == 1

    def test_not_default_values(self):
        options = Options(initial_k = 2, dm_case = 3,  kmeans_random_state = 0, kmeans_method = 2)
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
        assert options.distance_method == 1
        assert options.kmeans_method == 2

    def test_modify_attribute(self):
        with pytest.raises(FrozenInstanceError):
            options = Options()
            options.dm_case = 3

    def test_add_attribute(self):
        with pytest.raises(FrozenInstanceError):
            options = Options()
            options.dm_case2 = 2

    def test_preset_fail(self):
        with pytest.raises(ValueError):
            Options.preset("UnexistingLabel")

    def test_preset_success(self):
        assert Options.preset("opt1s0") == Options(p_parameter=0.01, dm_case=1, s_parameter=0.1, Standardize=0)
        assert Options.preset("opt2s0") == Options(p_parameter=0.01, dm_case=1, s_parameter=0.3, Standardize=0)
        assert Options.preset("opt12s0") == Options(p_parameter=0.05, dm_case=2, s_parameter=0.5, Standardize=0)
        assert Options.preset("opt11s1") == Options(p_parameter=0.05, dm_case=2, s_parameter=0.3, Standardize=1)
        assert Options.preset("opt12s1") == Options(p_parameter=0.05, dm_case=2, s_parameter=0.5, Standardize=1)

    def test_default_from_str(self):
        #creating a default Options object instance
        options = Options()

        #forcing to get str representation from Options object
        strOpt = options.__str__();

        #creating a new Options instance from
        optionsLoaded = Options.from_str(strOpt)

        #verifying if the object load and the original one are equals
        assert options == optionsLoaded


    def test_custom_from_str(self):
        #creating a custom Options object instance
        options = Options(initial_k = 2, dm_case = 3,  kmeans_random_state = 0, kmeans_method = 2)

        #forcing to get str representation from Options object
        strOpt = options.__str__();

        #creating a new Options instance from
        optionsLoaded = Options.from_str(strOpt)

        #verifying if the object load and the original one are equals
        assert options == optionsLoaded
