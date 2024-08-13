import numpy as np

def load_result(resultFilePath):

    loaded_data = np.load(resultFilePath, allow_pickle=True)
    dataset_results = loaded_data['dataset_results'].tolist()

    return dataset_results