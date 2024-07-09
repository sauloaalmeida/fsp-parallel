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

def verify_accuracy_fsp_from_optlabels(dataset_filenames=None, opt_labels=None , k_fold=None, number_of_runs=10, input_dataset_path=Path('.'), output_report_path=Path('.')):

    opts = {}

    if opt_labels != None:
        for opt_label in opt_labels:
            opts[opt_label] = Options.preset(opt_label)

    return verify_accuracy_fsp(dataset_filenames, opts, k_fold, number_of_runs, input_dataset_path, output_report_path)

def verify_accuracy_fsp(dataset_filenames=None, opts=None, k_fold=None, number_of_runs=10, input_dataset_path=Path('.'), output_report_path=Path('.')):
    """
    Evaluate the accuracy of FSP using cross-validation on different datasets and options.

    Parameters:
    dataset_filenames (list of str): List of dataset file names (default: predefined datasets)
    opt_labels (list of str): List of option labels (default: 'opt1s0' to 'opt12s1')
    k_fold (int): Number of folds for cross-validation (default: 0, which runs leave-one-out)
    number_of_runs (int): Number of runs for the evaluation (default: 30)

    Returns:
    List of dict containing mean accuracy, standard deviation, and total elapsed time for each dataset and option.
    """

    # Default dataset filenames if not provided
    if not dataset_filenames:
        dataset_filenames = [
            "Iris",
            "Bands",
            "DiabetesRisk",
            "Ionosphere",
            "Muskvs1",
            "Sonar",
            "ElectricalFaultDetection_2001Sample",
            "ElectricalFaultClassification_2000Sample"
        ]

    # Default option labels if not provided
    if not opts:
        for i in range(1, 13):
            for j in range(2):
                opts[f"opt{i}s{j}"] = Options.preset(f"opt{i}s{j}")

    # Initialize list to store results
    fsp_results = []

    # Iterate through the datasets
    for dataset_filename in dataset_filenames:
        filePath = input_dataset_path / "Datasets" / f"{dataset_filename}.csv"
        print(f"Input File Path: {str(filePath)}")
        data = pd.read_csv(filePath,header=None).values
        data[:, -1] -= 1 # Adjust class labels

        # Iterate through option labels
        for opt_label, opt in opts.items():
            accuracies_predict1 = []
            accuracies_predict2 = []
            elapsed_times = []

            # Perform the runs
            for _ in range(number_of_runs):
                # Set up the cross-validation partition
                cv = LeaveOneOut() if not k_fold else StratifiedKFold(n_splits=k_fold)

                # Train and evaluate the model
                acc1, acc2, elapsed_time = evaluate_model(data, cv, opt)
                accuracies_predict1.append(acc1)
                accuracies_predict2.append(acc2)
                elapsed_times.append(elapsed_time)

            # Store the results
            result = {
                'DatasetFileName': dataset_filename,
                'optLabel': opt_label,
                'MeanStd_accuracies_predict1': 100 * np.array([np.mean(accuracies_predict1), np.std(accuracies_predict1)]),
                'MeanStd_accuracies_predict2': 100 * np.array([np.mean(accuracies_predict2), np.std(accuracies_predict2)]),
                'KFold': data.shape[0] if not k_fold else k_fold,
                'NumberOfRuns': number_of_runs,
                'ElapsedTime': np.sum(elapsed_times),
                'accuracies_predict1': accuracies_predict1,
                'accuracies_predict2': accuracies_predict2
            }
            fsp_results.append(result)

            # Display fsp_results as a DataFrame
            print(pd.DataFrame(fsp_results).tail(1))

            # Save fsp_results to a csv file
            filePath = output_report_path / "Results" / f"FSP_{' '.join(dataset_filenames)}_KFold{k_fold}_NumberOfRuns{number_of_runs}.csv"
            pd.DataFrame(fsp_results).to_csv(filePath, index=False)

    return fsp_results, str(filePath)

def evaluate_model(data, cv, opt):
    """
    Evaluate a model using cross-validation.

    Parameters:
    data (array-like): Dataset (N x (d+1) numpy array where the first d columns are X and the last column is y)
    cv (cross-validation object): Cross-validation object (e.g., StratifiedKFold, LeaveOneOut)
    opt (Options object): Options for the model

    Returns:
    tuple: Mean accuracy for prediction 1, mean accuracy for prediction 2, and elapsed time for the evaluation.
    """
    indices = list(cv.split(data[:, :-1], data[:, -1]))

    start_time = time.time()
    # Run evaluation in parallel and collect results
    results = Parallel(n_jobs=-1)(delayed(evaluate_model_single_fold)(data, train_index, test_index, opt)
                        for train_index, test_index in indices)
    elapsed_time = time.time() - start_time

    # Extract accuracies from results
    acc1 = np.array([result[0] for result in results])
    acc2 = np.array([result[1] for result in results])

    return np.mean(acc1), np.mean(acc2), elapsed_time

def evaluate_model_single_fold(data, train_index, test_index, opt):
    """
    Train and evaluate a model for a single fold of cross-validation.
    """
    X_train, X_test = data[train_index, :-1], data[test_index, :-1]
    y_train, y_test = data[train_index, -1].astype(int), data[test_index, -1].astype(int)

    mdl = fsp(X_train, y_train, opt)

    y_pred1, _ = fsp_predict(mdl, X_test,1)
    acc1 = np.mean(y_test == y_pred1)

    y_pred2, _ = fsp_predict(mdl, X_test,2)
    acc2 = np.mean(y_test == y_pred2)

    return acc1, acc2