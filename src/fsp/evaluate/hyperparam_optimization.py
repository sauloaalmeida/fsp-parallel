import numpy as np
import pandas as pd
import itertools
from pathlib import Path
from fsp.options import Options
from fsp.evaluate.fspaccuracy import FSP_cvLoss
from sklearn import datasets
from sklearn.model_selection import train_test_split, LeaveOneOut
from joblib import Parallel, delayed
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.plots import plot_convergence
from skopt.utils import use_named_args
import os
import csv

def fsp_HyperparameterTuning_skopt(dataset_name):
    """
    Perform hyperparameter tuning for the FSP model using scikit-optimize.

    Parameters:
    dataset_name (stror numpy.ndarray): The input dataset.
                                        If a string is provided, it is treated as the name of CSV file located in the "Datasets" folder.
                                        The dataset should have features in columns and the target variable in the last column.
                                        If a numeric array is provided, it represents the dataset itself.

    Returns:
    opt (Options): The optimized FSP options.
    predict_method (int): The best prediction method.
    accuracy (float): The accuracy of the model on the test set.
    OptimizeResult: The result of the optimization process.
    """
    absPath = Path(".").cwd() / "test" / "benchmark" / "fsp" / "Datasets" / f"{dataset_name}.csv"

    # Load [X,y] matrix based on the dataset_name type (str or numpy.ndarray)
    if isinstance(dataset_name, str):
        X_y = pd.read_csv(absPath,header=None).values
    elif isinstance(dataset_name, np.ndarray):
        X_y = dataset_name
        dataset_name = None
    else:
        raise ValueError("dataset_name must be a string or a numpy.ndarray")

    # Adjust class labels if necessary
    if np.min(np.unique(X_y[:, -1])) == 1:
        X_y[:, -1] -= 1

    # Split dataset into features (X) and target (y)
    X, y = X_y[:, :-1], X_y[:, -1].astype(int)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # Combine X_train and y_train for cross-validation
    X_y_train = np.hstack((X_train, y_train.reshape(-1, 1)))

    # Define the objective function for hyperparameter optimization
    def objective(params):
        Standardize, initial_k, p_parameter, h_threshold, dm_case, s_parameter, dm_threshold, update_s_parameter, predict_method = params
        # Create an options object with the suggested hyperparameters
        opt = Options(
            Standardize=Standardize,
            initial_k=initial_k,
            p_parameter=p_parameter,
            h_threshold=h_threshold,
            dm_case=dm_case,
            s_parameter=s_parameter,
            dm_threshold=dm_threshold,
            update_s_parameter=update_s_parameter
        )
        # Run evaluation in parallel and collect results in single_run
        single_run = Parallel(n_jobs=-1)(
            delayed(FSP_cvLoss)(X_y_train, opt, predict_method=int(predict_method), cv=LeaveOneOut())
            for _ in range(10)
        )
        # Return the objective value for optimization
        return np.mean(single_run) + 2 * np.std(single_run)

    # Define the space of hyperparameters to search
    kmax = int( 10 + (90/9900)*(max(100,X.shape[0])-100) )
    print(f"\nintial_k space search = {[1,kmax]}\n")
    space = [
        Integer(0, 1,  name='Standardize'),
        Integer(1,kmax,name='initial_k'),
        Real(0.0, 0.1, name='p_parameter'),
        Integer(1, 10, name='h_threshold'),
        Integer(1, 2,  name='dm_case'),
        Real(0.1, 0.5, name='s_parameter'),
        Integer(3, 10, name='dm_threshold'),
        Integer(0, 1,  name='update_s_parameter'),
        Integer(1, 2,  name='predict_method')
    ]

    # Define initial parameter combinations
    # Case 1: If the file "Results/{dataset_name}_fsp_evaluating_accuracy_KFold0.npz" exist
    if os.path.isfile(f"Results/{dataset_name}_fsp_evaluating_accuracy_KFold0.npz"):
        loaded_data = np.load(f"Results/{dataset_name}_fsp_evaluating_accuracy_KFold0.npz", allow_pickle=True)
        dataset_results = loaded_data['dataset_results'].tolist()
        x0 = []; y0 = []
        for R in dataset_results:
            opt = R['opt']
            x0.append([opt.Standardize, opt.initial_k, opt.p_parameter, opt.h_threshold, opt.dm_case, opt.s_parameter, opt.dm_threshold, opt.update_s_parameter, 1])
            x0.append([opt.Standardize, opt.initial_k, opt.p_parameter, opt.h_threshold, opt.dm_case, opt.s_parameter, opt.dm_threshold, opt.update_s_parameter, 2])
            y0.extend([1-R['MeanStd_accuracies_predict1'][0]/100+2*R['MeanStd_accuracies_predict1'][1]/100, 1-R['MeanStd_accuracies_predict2'][0]/100+2*R['MeanStd_accuracies_predict2'][1]/100])
    # Case 2: If the file "Results/{dataset_name}_fsp_evaluating_accuracy_KFold0.npz" doesn't exist
    else:
        # Define initial parameter values for each variable
        param_values = {
            'Standardize': [0, 1],
            'initial_k': [1, 5, 10],
            'p_parameter': [0.01],
            'h_threshold': [1, 10],
            'dm_case': [1, 2],
            's_parameter': [0.1],
            'dm_threshold': [3, 5],
            'update_s_parameter': [0, 1],
            'predict_method': [1, 2]
        }
        # Generate combinations of initial parameter values
        combinations = list(itertools.product(
            param_values['Standardize'],
            param_values['initial_k'],
            param_values['p_parameter'],
            param_values['h_threshold'],
            param_values['dm_case'],
            param_values['s_parameter'],
            param_values['dm_threshold'],
            param_values['update_s_parameter'],
            param_values['predict_method']
        ))
        # Convert initial parameter combinations to the format required by skopt
        x0 = [list(combo) for combo in combinations]
        y0 = None

    # Perform the optimization
    print(f"x0={x0}")
    print(f"y0={y0}\n")
    # len_x0 = len(x0) if y0 is None else 0
    len_x0 = 0
    # n_initial_points = 10
    # n_calls = 10
    n_initial_points = 2
    n_calls = 2
    OptimizeResult = gp_minimize(
        func=objective,
        dimensions=space,
        # x0=x0,
        # y0=y0,
        n_initial_points=n_initial_points,
        n_calls=len_x0 + n_initial_points + n_calls,
        n_jobs=-1,
        verbose=True
    )

    # Plot the convergence history
    plot_convergence(OptimizeResult)

    # Print the function value at the minimum
    max_best_value = 100*(1 - OptimizeResult.fun)
    print("\nMinimum function value:", OptimizeResult.fun)

    # Retrieve the best parameters from the optimizer
    best_params = OptimizeResult.x
    print("\nBest parameters:", best_params)

    # Separate the predict_method from the best parameters
    predict_method = int(best_params.pop(-1))

    # Create an options object with the best parameters
    opt = Options(
        Standardize=int(best_params[0]),
        initial_k=int(best_params[1]),
        p_parameter=best_params[2],
        h_threshold=int(best_params[3]),
        dm_case=int(best_params[4]),
        s_parameter=best_params[5],
        dm_threshold=int(best_params[6]),
        update_s_parameter=int(best_params[7])
    )
    print("\nOptimized options:")
    print(opt)

    # Avaliate the model with the optimized options
    vecAcc = np.zeros(10)
    # for i in range(10):
    #     # Train the model with the optimized options
    #     mdl = fsp(X_train, y_train, opt)
    #     # Predict the test set using the trained model
    #     ypred, _ = fsp_predict(mdl, X_test, predict_method)
    #     # Calculate the accuracy of the model
    #     vecAcc[i] = 100*np.mean(y_test == ypred)
    # Train and evaluate the model in parallel using a lambda function
    vecAcc = Parallel(n_jobs=-1)(
        delayed(lambda i: 100 * np.mean(
            y_test == fsp_predict(fsp(X_train, y_train, opt), X_test, predict_method)[0]
        ))(i) for i in range(10)
    )

    # Calculate and print the accuracy of the model
    MeanStd_accuracies = [np.mean(vecAcc), np.std(vecAcc)]
    print(f'\nMeanStd_accuracies: {MeanStd_accuracies}')

    # Save results in the "Results" folder if the input "dataset_name" is of type str
    if isinstance(dataset_name, str):
        # Create the "Results" folder if it doesn't exist
        results_dir = "Results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # Define the result file name
        result_file = os.path.join(results_dir, f"{dataset_name}_fsp_HyperparameterTuning.csv")

        # Check if the file already exists
        file_exists = os.path.isfile(result_file)

        # Save opt, predict_method, accuracy, and OptimizeResult in a CSV file
        with open(result_file, mode='a', newline='') as f:
            writer = csv.writer(f)

            # Write header only if the file doesn't already exist
            if not file_exists:
                writer.writerow(['optimizer', 'opt', 'predict_method', 'max_best_value', 'MeanStd_accuracies'])

            # Write the results
            writer.writerow(["skopt", opt, predict_method, max_best_value, MeanStd_accuracies])

    # Return results
    return opt, predict_method, max_best_value, MeanStd_accuracies, OptimizeResult