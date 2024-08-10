import time
import os
import numpy as np
import pandas as pd
from pathlib import Path
from fsp.options import Options
from fsp.fsp import fsp
from fsp.fsp import fsp_predict
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from joblib import Parallel, delayed

#################################################
#### FSP CVLOSS ALGORITHM BLOCK BEGIN
#################################################

def FSP_cvLoss(data, opt=Options, predict_method=[1,2], cv=LeaveOneOut()):
    """
    Evaluate the FSP model using cross-validation.

    Parameters:
    data (array-like): Dataset N x (d+1) numpy array where the first d columns are X and the last column is y.
    opt (Options object): Options for the model.
    predict_method (int or list of int): Prediction method(s) to be evaluated.
    cv (cross-validation object): Cross-validation object (e.g., StratifiedKFold, LeaveOneOut).

    Returns:
    float or array-like: Cross-validation error for each prediction method given in 'predict_method'.
    """
    # Check if predict_method is an integer; if so, convert it to a list and set single_method to True
    single_method = False
    if isinstance(predict_method, int):
        predict_method = [predict_method]
        single_method = True

    # Split the data according to the cross-validation strategy
    indices = list(cv.split(data[:, :-1], data[:, -1]))

    # Run evaluation in parallel and collect results in single_fold_loss
    single_fold_loss = Parallel(n_jobs=-1)(
        delayed(FSP_cvLoss_SingleFold)(data, opt, predict_method, train_index, test_index)
        for train_index, test_index in indices)

    # Calculate the mean loss across all folds
    kfold_loss = np.mean(single_fold_loss, axis=0)

    # If only one prediction method was provided, return a single float value
    if single_method:
        return kfold_loss[0]

    return kfold_loss

def FSP_cvLoss_SingleFold(data, opt, predict_method, train_index, test_index):
    """
    Train and evaluate the FSP model for a single fold of cross-validation.
    """
    # Split the data into training and testing sets
    X_train, X_test = data[train_index, :-1], data[test_index, :-1]
    y_train, y_test = data[train_index, -1].astype(int), data[test_index, -1].astype(int)

    # Train the model
    mdl = fsp(X_train, y_train, opt)

    # Initialize loss vector
    single_fold_loss = np.zeros(len(predict_method))

    # Evaluate the model using specified prediction methods
    for j, method in enumerate(predict_method):
        y_pred, _ = fsp_predict(mdl, X_test, method)
        single_fold_loss[j] = np.mean(y_test != y_pred)

    return single_fold_loss

#################################################
#### FSP CVLOSS ALGORITHM BLOCK END
#################################################


#################################################
#### FSP ACCURACY ALGORITHM BLOCK BEGIN
#################################################

def fsp_evaluating_accuracy(dataset_name=None, opt=Options.list1(), KFold=0, NumberOfRuns=10):
    """
    Evaluate the accuracy of FSP using cross-validation.

    Parameters:
    dataset_name (str, list of str or numpy.ndarray): The input dataset(s).
                                                      If a string or string array is provided, it is treated as the name(s) of CSV file(s) located in the "Datasets" folder.
                                                      Each dataset should have features in columns and the target variable in the last column.
                                                      If a numeric array is provided, it represents the dataset itself.
    opt (Options or list of Options): Options for the FSP algorithm.
    KFold (int): Number of folds for cross-validation (default: 0, which runs leave-one-out).
    NumberOfRuns (int): Number of runs for the evaluation (default: 10).

    Returns:
    List of dict containing mean accuracy, standard deviation, and total elapsed time for each dataset_name and option.
    """

    projectRootAbsPath = Path(".").cwd()
    datasetAbsDirPath = projectRootAbsPath / "test" / "benchmark" / "fsp" / "Datasets"
    resultsAbsDirPath = projectRootAbsPath / "test" / "benchmark" / "fsp" / "Results"

    # Define default dataset_name, if not informe
    if not dataset_name:
        dataset_name = [
        "Iris",
        "Bands",
        "DiabetesRisk",
        "GlassIdentification",
        "Ionosphere",
        "LibrasMovement",
        "MaternalHealthRisk",
        "Sonar",
        "Zoo",
        "ElectricalFaultDetection_2001Sample"
        ]

    # Load [X,y] matrix based on the dataset_name type (str, list of str, or numpy.ndarray)
    if all( isinstance(ds, str) for ds in dataset_name ):
        list_X_y = [pd.read_csv(datasetAbsDirPath / f"{ds}.csv", header=None).values for ds in dataset_name]
    elif isinstance(dataset_name, np.ndarray):
        list_X_y = [dataset_name]
    else:
        raise ValueError("Invalid dataset_name type. Must be str, list of str, or numpy.ndarray.")

    # Verify opt
    if not all(isinstance(opt_j, Options) for opt_j in opt):
        raise ValueError("Invalid opt type. Must be Options object or list of Options object.")

    # Initialize list to store results
    results = []

    # Iterate through the list_X_y
    for i, X_y in enumerate(list_X_y):
        # Define current dataset name
        if isinstance(dataset_name, np.ndarray):
            current_dataset_name = None
        else:
            current_dataset_name = dataset_name if isinstance(dataset_name, str) else dataset_name[i]
            print(f"Current dataset name: {current_dataset_name}")

        # Adjust class labels if necessary
        if np.min(np.unique(X_y[:, -1])) == 1:
            X_y[:, -1] -= 1

        # Initialize list to store the dataset results
        dataset_results = []

        # Make dir Results, if it doesn's exist
        os.makedirs('Results', exist_ok=True)

        # Define the dataset results file name
        dataset_results_FullFileName = resultsAbsDirPath / f"{current_dataset_name}_fsp_evaluating_accuracy_KFold{KFold}.npz"

        # Check if the file already exists
        if current_dataset_name and os.path.isfile(dataset_results_FullFileName):
            loaded_data = np.load(dataset_results_FullFileName, allow_pickle=True)
            dataset_results = loaded_data['dataset_results'].tolist()

        # Iterate through opt
        for current_opt in opt:
            # Check if the current opt has already been executed
            existing_result_idx = [i for i, r in enumerate(dataset_results) if r['opt'] == current_opt]
            if len(existing_result_idx) > 0:
                existing_result = dataset_results[existing_result_idx[0]]

                # Check if the existing result has already been executed at least the current value of the input 'NumberOfRuns'
                if existing_result['NumberOfRuns'] >= NumberOfRuns:
                    print(f"\nThe following option has already been executed for dataset {current_dataset_name}. Skipping...")
                    print(current_opt)
                    results.append(existing_result)
                    print(pd.DataFrame(results).tail(1))
                    continue
                # Reuse existing results and perform remaining runs
                else:
                    remaining_runs = NumberOfRuns - existing_result['NumberOfRuns']
                    print(f"\nThe following option has been partially executed for dataset {current_dataset_name}. Running the remaining {remaining_runs} executions.")
                    print(current_opt)
                    accuracies_predict1 = np.concatenate((existing_result['accuracies_predict1'], np.zeros(remaining_runs)))
                    accuracies_predict2 = np.concatenate((existing_result['accuracies_predict2'], np.zeros(remaining_runs)))
                    start_run = existing_result['NumberOfRuns']
                    ElapsedTime = existing_result['ElapsedTime']
            else:
                accuracies_predict1 = np.zeros(NumberOfRuns)
                accuracies_predict2 = np.zeros(NumberOfRuns)
                start_run = 0
                ElapsedTime = 0

            # Iterate through NumberOfRuns
            tStart = time.time()
            for run in range(start_run, NumberOfRuns):
                # Set up the cross-validation partition
                cv = LeaveOneOut() if not KFold else StratifiedKFold(n_splits=KFold)
                # Train and evaluate the model
                kfold_loss = FSP_cvLoss(X_y, current_opt, [1, 2], cv)
                accuracies_predict1[run] = 100*(1 - kfold_loss[0])
                accuracies_predict2[run] = 100*(1 - kfold_loss[1])
            # Calculate elapsed time
            elapsed_time = time.time() - tStart

            # Create a dict with the current result
            result = {
                'dataset_name': current_dataset_name,
                'opt': current_opt,
                'MeanStd_accuracies_predict1': np.array([np.mean(accuracies_predict1), np.std(accuracies_predict1)]),
                'MeanStd_accuracies_predict2': np.array([np.mean(accuracies_predict2), np.std(accuracies_predict2)]),
                'KFold': KFold,
                'NumberOfRuns': NumberOfRuns,
                'ElapsedTime': elapsed_time,
                'accuracies_predict1': accuracies_predict1,
                'accuracies_predict2': accuracies_predict2
            }

            # Store the current result in dataset_results
            if len(existing_result_idx) > 0:
                dataset_results[existing_result_idx[0]] = result
            else:
                dataset_results.append(result)

            # Store the current result in results
            results.append(result)

            # Save the dataset results if the dataset filename is known
            if isinstance(current_dataset_name, str):
                np.savez(dataset_results_FullFileName, dataset_results=dataset_results)

            # Display results as a DataFrame
            print(pd.DataFrame(results).tail(1))

    # Return results
    return results

#################################################
#### FSP ACCURACY ALGORITHM BLOCK BEGIN
#################################################