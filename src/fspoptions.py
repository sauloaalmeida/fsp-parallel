class FspOptions:

    __initial_k
    __p_parameter
    __h_threshold
    __dm_case
    __dm_threshold
    __update_dm_threshold
    __return_full_dm
    __return_full_history
    __iteration_threshold
    __kmeans_random_state

    def __init__(self,
                 initial_k=1,
                 p_parameter=0.05,
                 h_threshold=0,
                 dm_case=2,
                 dm_threshold=0.5,
                 update_dm_threshold=True,
                 return_full_dm=False,
                 return_full_history=False,
                 iteration_threshold=2e6,
                 kmeans_random_state=None):
        self.__initial_k = initial_k
        self.__p_parameter = p_parameter
        self.__h_threshold = h_threshold
        self.__dm_case = dm_case
        self.__dm_threshold = dm_threshold
        self.__update_dm_threshold = update_dm_threshold
        self.__return_full_dm = return_full_dm
        self.__return_full_history = return_full_history
        self.__iteration_threshold = iteration_threshold
        self.__kmeans_random_state = kmeans_random_state

    def __str__(self):
        return str(vars(self))

    @property
    def initial_k(self):
        return self.__initial_k

    @property
    def p_parameter(self):
        return self.__

    @property
    def h_threshold(self):
        return self.__p_parameter

    @property
    def dm_case(self):
        return self.__dm_case

    @property
    def dm_threshold(self):
        return self.__dm_threshold

    @property
    def update_dm_threshold(self):
        return self.__update_dm_threshold

    @property
    def return_full_dm(self):
        return self.__return_full_dm

    @property
    def return_full_history(self):
        return self.__return_full_history

    @property
    def iteration_threshold(self):
        return self.__iteration_threshold

    @property
    def kmeans_random_state(self):
        return self.__kmeans_random_state

    def helloFsp(self):
        return "hello fsp"