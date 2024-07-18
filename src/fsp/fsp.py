import time
import numpy as np
import pandas as pd
from fsp.options import Options
from fsp.divergence_measure.dm_algorithm import Divergence_Measure

#################################################
#### PREDICT BLOCK BEGIN
#################################################

def fsp_predict(fsp_output, Xtest, predict_method=1, opt=Options):
    """Predict labels using the Feature Space Partition (FSP) method's output.

    Parameters:
        fsp_output (dict): Dictionary containing the FSP method's output.
        Xtest (array-like): Test dataset.
        predict_method (int, optional): Method to use for prediction (1 or 2). Default is 1.

    Returns:
        tuple: Predicted class labels and their proportions.
    """

    # Standardize Xtest if required
    if fsp_output['opt'].Standardize:
        Xtest = (Xtest - fsp_output['Mu']) / fsp_output['Sigma']

    if predict_method == 1:
        return predict_method_1(fsp_output, Xtest, opt)
    elif predict_method == 2:
        return predict_method_2(fsp_output, Xtest, opt)
    else:
        raise ValueError("Invalid predict_method. Must be 1 or 2.")

def predict_method_1(fsp_output, Xtest, opt):
    """Prediction method 1 using nearest centroids."""
    C = fsp_output['C']  # k x d matrix of cluster centroids

    # Classify the training data
    Itrain = classify_by_the_nearest_centroid(fsp_output['Xtrain'], C, opt)

    # Calculate the dominant class label and its proportion for each cluster
    DominantClassLabel, DominantClassLabel_Proportion = calculate_dominant_class(Itrain, fsp_output['ytrain'], C.shape[0], fsp_output['ClassNames'].size)

    # Classify the test data
    Itest = classify_by_the_nearest_centroid(Xtest, C, opt)

    # Assign the predicted class label and its proportion
    ypredict = DominantClassLabel[Itest]
    ypredict_Proportion = DominantClassLabel_Proportion[Itest]

    return ypredict, ypredict_Proportion

def predict_method_2(fsp_output, Xtest, opt):
    """Prediction method 2 using hierarchical clustering."""
    # Get the number of test observations
    Ntest = Xtest.shape[0]

    if Ntest == 1:
        ypredict, ypredict_Proportion = run_single_prediction(Xtest, fsp_output['H'], opt )
    else:
        ypredict, ypredict_Proportion = run_multi_prediction(Xtest, fsp_output['H'], opt)

    return ypredict, ypredict_Proportion

def classify_by_the_nearest_centroid(X, C, opt):
    """Classify data points by the nearest centroid."""
    dist_X_C = opt.getDistanceMethod().cdist(X, C)
    return np.argmin(dist_X_C, axis=1)

def calculate_dominant_class(Itrain, ytrain, k, nCL):
    """Calculate the dominant class label and its proportion for each cluster."""
    count_matrix = np.zeros((k, nCL), dtype=int)
    np.add.at(count_matrix, (Itrain, ytrain), 1)

    DominantClassLabel = np.argmax(count_matrix, axis=1)
    DominantClassLabel_Proportion = count_matrix[np.arange(k), DominantClassLabel] / np.sum(count_matrix, axis=1)

    return DominantClassLabel, DominantClassLabel_Proportion

def run_single_prediction(Xtest_row, H, opt):
    """Predict the class and proportion for a single test observation."""

    # Iterate through H
    veci = H['i']
    for j in range(len(veci)):
        if len(H['rmCidx'][j]) == 0:
            continue

        # Classify the test data
        # dists = np.linalg.norm(Xtest_row - H['C'][j], axis=1)
        # Itest = np.argmin(dists)
        Itest = classify_by_the_nearest_centroid(Xtest_row, H['C'][j], opt)

        # Check if Xtest_row belongs to a cluster marked for removal
        vecbool = np.isin(H['rmCidx'][j],Itest)
        idx_in_rmCidx = np.where(vecbool)[0]

        # If Xtest_row belongs to a cluster marked for removal, return
        if len(idx_in_rmCidx) > 0:
            ypredict = H['rmDominantClassLabel'][j][idx_in_rmCidx]
            ypredict_Proportion = H['rmDominantClassLabel_Proportion'][j][idx_in_rmCidx]
            return ypredict, ypredict_Proportion

    raise ValueError('Did not find an iteration in H where the observation Xtest_row belongs to a cluster marked for removal.')

def run_multi_prediction(Xtest, H, opt):
    raise NotImplementedError('predict_method_2 with Xtest.shape[0]>1 is not implemented yet.')

#################################################
#### PREDICT BLOCK END
#################################################


#################################################
#### CHECK HOMOGENEITY BLOCK BEGIN
#################################################
def check_homogeneity(y, idx, k:int, nCL:int, p_parameter:float, h_threshold:int) -> tuple:
    """
    Evaluate the homogeneity of clusters based on class label distribution within each cluster.

    Parameters:
    y (numpy.ndarray): m-element vector where each entry contains the label of the corresponding observation. Labels are integers between 0 and nCL-1 (nCL is the number of unique class labels).
    idx (numpy.ndarray): m-element vector where each entry contains the index of the nearest centroid for the corresponding observation, as determined by the k-means method.
    k (int): Number of clusters.
    nCL (int): Number of unique class labels.
    p_parameter (float): The p-almost homogeneous parameter, a threshold for determining homogeneity.
    h_threshold (int): Minimum number of observations required for a cluster to be considered in the homogeneity test.

    Returns:
    tuple: (rmCidx, rmXidx, DominantClassLabel, DominantClassLabel_Proportion, ObservationsPerCluster, nonDominantClassLabelNumber, count_matrix)
        - rmCidx (numpy.ndarray): Indices of clusters that satisfy the homogeneity criterion.
        - rmXidx (numpy.ndarray): Indices of observations in clusters that satisfy the homogeneity criterion.
        - DominantClassLabel (numpy.ndarray): k-element vector with the dominant class label for each cluster.
        - DominantClassLabel_Proportion (numpy.ndarray): k-element vector with the proportion of the dominant class in each cluster.
        - ObservationsPerCluster (numpy.ndarray): k-element vector with the number of observations in each cluster.
        - nonDominantClassLabelNumber (numpy.ndarray): k-element vector with the number of observations that do not belong to the dominant class in each cluster.
        - count_matrix (numpy.ndarray): k x nCL matrix with the count of occurrences of each class in each cluster.
"""
    # Step 0: Count the occurrences of each class in each cluster
    count_matrix = np.zeros((k, nCL), dtype=int)  # Initialize a k x n matrix to count occurrences of each class in each cluster
    np.add.at(count_matrix, (idx, y), 1)  # Count the occurrences of each class in each cluster

    # Step 1: Calculate the dominant class label and its proportion for each cluster
    DominantClassLabel = np.argmax(count_matrix, axis=1)                                # k-element vector. Dominant class label in each cluster
    DominantClassLabel_Counts = count_matrix[np.arange(k), DominantClassLabel]          # k-element vector. Number of observations in each cluster that belong to the dominant class
    ObservationsPerCluster = np.sum(count_matrix, axis=1)                               # k-element vector. Total number of observations in each cluster
    DominantClassLabel_Proportion = DominantClassLabel_Counts / ObservationsPerCluster  # k-element vector. Proportion of the dominant class in each cluster

    # Step 2: Calculate the number of observations that do not belong to the dominant class in each cluster
    nonDominantClassLabelNumber = ObservationsPerCluster - DominantClassLabel_Counts

    # Step 3: Determine which clusters satisfy the homogeneity criterion
    # Find clusters where the dominant class proportion is at least (1 - p_parameter) and the number of observations is at least h_threshold
    rmCidx = np.where( (DominantClassLabel_Proportion >= (1 - p_parameter)) & (ObservationsPerCluster >= h_threshold) )[0]

    # Step 4: Determine which observations are in clusters that satisfy the homogeneity criterion
    if len(rmCidx) > 0:
        rmXidx = np.where(np.isin(idx, rmCidx))[0]
    else:
        rmXidx = np.array([])

    return rmCidx, rmXidx, DominantClassLabel, DominantClassLabel_Proportion, ObservationsPerCluster, nonDominantClassLabelNumber, count_matrix

#################################################
#### CHECK HOMOGENEITY BLOCK END
#################################################

#################################################
#### CHECK SEPARABILTY BLOCK BEGIN
#################################################
def check_separability(X, y, idx, k:int, nCL:int, count_matrix, dm_case:int, s_parameter:float, dm_threshold:int, return_full_dm:bool, opt:Options) -> tuple:
    """
    Check the separability of clusters based on divergence measures.

    Parameters:
    X (numpy.ndarray): m x d matrix where each row contains the d-dimensional coordinates of an observation (m is the number of observations, d is the number of features).
    y (numpy.ndarray): m-element vector where each entry contains the label of the corresponding observation. Labels are integers between 0 and nCL-1 (nCL is the number of unique class labels).
    idx (numpy.ndarray): m-element vector where each entry contains the index of the nearest centroid for the corresponding observation, as determined by the k-means method.
    k (int): Number of clusters.
    nCL (int): Number of unique class labels.
    count_matrix (numpy.ndarray): k x nCL matrix where the entry (c, l) contains the number of occurrences of class l in cluster c.
    dm_case (int): It specifies the method to be used in the Divergence_Measure function.
    s_parameter (float): It is the separability parameter. For a cluster to be considered s-separable, there must be at least one pair of distinct class labels, a and b, such that the divergence measure between the observations within the cluster with labels a and b is at least s.
    dm_threshold (int): This variable's role is to restrict the use of the Divergence_Measure function. For a given cluster and class labels a and b, the divergence measure will only be calculated with the Divergence_Measure function if the number of observations with class a and class b have each at least dm_threshold units. If this condition is not met, we assign the divergence measure as zero.
    return_full_dm (bool): If True, the full divergence measure matrix is calculated. Otherwise, return when the separability criterion is met.

    Returns:
    tuple: (criterion_met, mat_dm)
        - criterion_met (bool): True if the separability criterion is met, False otherwise.
        - mat_dm (numpy.ndarray): k x (nCL*(nCL-1)/2) matrix of divergence measures.
    """

    # Initialize variables
    criterion_met = False
    mat_dm = np.zeros((k,nCL*(nCL-1)//2))

    # Clusters that have at least two classes, with each class containing at least dm_threshold observations.
    matBool = count_matrix >= dm_threshold # Boolean matrix where True if class count >= dm_threshold
    vecBool = np.sum(matBool, axis=1) >= 2 # Boolean vector where True if cluster has at least two classes meeting threshold
    search_clusters = np.where(vecBool)[0] # Indices of clusters to be searched

    # Compute the divergence measure
    for c in search_clusters:
        idx_c = idx == c  # Boolean index for observations in cluster c
        X_c = X[idx_c,:]  # Observations in cluster c
        y_c = y[idx_c]    # Class labels of observations in cluster c

        for a in range(0,nCL-1):
            Na = count_matrix[c,a]         # Number of observations in X_c with class a
            if Na < dm_threshold: continue # Skip if class a does not meet minimum count of dm_threshold units
            X_c_a = X_c[y_c == a,:]        # Observations in X_c with class label a

            for b in range(a+1,nCL):
                Nb = count_matrix[c,b]         # Number of observations in X_c with class b
                if Nb < dm_threshold: continue # Skip if class b does not meet minimum count of dm_threshold units
                dm = Divergence_Measure(X_c_a, X_c[y_c == b,:], opt) # Calculate divergence measure
                mat_dm[c,a*(nCL-1)-a*(a-1)//2+b-a-1] = dm
                if dm >= s_parameter:
                    criterion_met = True # Set criterion met to True if divergence measure meets or exceeds s_parameter
                    if not return_full_dm:
                        return criterion_met, mat_dm # Return early if not required to compute full matrix
    return criterion_met, mat_dm
#################################################
#### CHECK SEPARABILTY BLOCK END
#################################################


#################################################
#### INITIALIZE HOSTORICAL DATA BLOCK BEGIN
#################################################
def Initialize_historical_data_structure():
    H = {
        'i': [],
        'k': [],
        's_parameter': [],
        'dm': [],
        'count_matrix': [],
        'ClassLabel_Proportion': [],
        'DominantClassLabel': [],
        'DominantClassLabel_Proportion': [],
        'rmCidx': [],
        'rmC': [],
        'rmDominantClassLabel': [],
        'rmDominantClassLabel_Proportion': [],
        'ClassificationError': [],
        'sum1': [],
        'C': [],
        'rmXidx': []
    }
    return H
#################################################
#### INITIALIZE HISTORICAL DATA BLOCK END
#################################################


#################################################
#### STORE ITERATION INFO BLOCK BEGIN
#################################################
def store_iteration_info(H, i, C, count_matrix, ObservationsPerCluster,
                         DominantClassLabel, DominantClassLabel_Proportion,
                         rmCidx, s_parameter, dm, ClassificationError, sum1, rmXidx):
    """
    Store iteration information in the history dictionary H.
    """
    H['i'].append(i)
    H['k'].append(C.shape[0])
    H['s_parameter'].append(s_parameter)
    H['dm'].append(dm)

    H['count_matrix'].append(count_matrix)
    H['ClassLabel_Proportion'].append(count_matrix / ObservationsPerCluster[:, np.newaxis])
    H['DominantClassLabel'].append(DominantClassLabel)
    H['DominantClassLabel_Proportion'].append(DominantClassLabel_Proportion)

    H['rmCidx'].append(rmCidx)
    H['rmC'].append(C[rmCidx, :])
    H['rmDominantClassLabel'].append(DominantClassLabel[rmCidx])
    H['rmDominantClassLabel_Proportion'].append(DominantClassLabel_Proportion[rmCidx])

    H['ClassificationError'].append(ClassificationError)
    H['sum1'].append(sum1)
    H['C'].append(C)
    H['rmXidx'].append(rmXidx)
#################################################
#### STORE ITERATION INFO BLOCK END
#################################################




#################################################
#### FSP ALGORITHM BLOCK BEGIN
#################################################
def fsp(X, y , opt = Options()):
    # Start measuring the total execution time
    startTime = time.time()

    # Check if class names are integers from 0 to nCL-1
    ClassNames = np.unique(y)
    nCL = len(ClassNames)
    if not np.all(np.arange(nCL) == ClassNames):
        raise ValueError("Class names must be integers from 0 to nCL-1, where nCL is the number of unique class labels.")

    # Initialize the historical data structure H to store iteration information
    H = Initialize_historical_data_structure()

    # Initializing some other additional variables
    k = opt.initial_k              # Current value of k for k-means
    i = -1                         # Iteration counter
    N = X.shape[0]                 # N is the number of observations in the input data X
    s_parameter = opt.s_parameter; # The separability parameter
    sum2 = 0                       # This variable is employed in the calculation of the classification error

    # Standardize X if specified in the options (opt.Standardize == true)
    if opt.Standardize:
        Mu = np.mean(X, axis=0)
        Sigma = np.std(X, axis=0)
        X = (X - Mu) / Sigma
    else:
        Mu = None
        Sigma = None

    # Store normalization parameters in the Output structure
    result = dict()
    result['Mu'] = Mu
    result['Sigma'] = Sigma

    # Store the dataset information in the Output structure to facilitate its use outside of this function
    result['Xtrain'] = X # Training data
    result['ytrain'] = y # Training labels
    result['ClassNames'] = ClassNames # Names of the classes

    # recovery the kmeans method that must be used
    kmeans_method = opt.getKMeansMethod()

    # While X not empty do
    while X.shape[0] > 0:
        # Increment iteration counter
        i += 1 # Let i refers to current iteration

        # Step 1: Segment X into k clusters using k-means
        idx, C = kmeans_method.kmeans(X=X, k=k, random_state=opt.kmeans_random_state)

        # Step 2: Evaluate segmentation
        # 2.1: Check homogeneity
        (rmCidx,
         rmXidx,
         DominantClassLabel,
         DominantClassLabel_Proportion,
         ObservationsPerCluster,
         nonDominantClassLabelNumber,
         count_matrix) = check_homogeneity(y, idx, k, nCL, opt.p_parameter, opt.h_threshold)

        # 2.2: Calculate the classification error
        ClassificationError = sum2 + np.sum(nonDominantClassLabelNumber) / N

        # 2.3: Update separability parameter?????????????????????
        #if i > 0 and opt.update_s_parameter and ClassificationError != 0 and ClassificationError_previous != 0:
        #    s_parameter = s_parameter / np.sqrt(ClassificationError / ClassificationError_previous)

        # Step 3: If there is any homogeneous region
        dm = np.array([])
        if len(rmCidx) > 0:
            # 3.1: If the number of observations in non-homogeneous regions is ≤ initial_k or the iteration has reached the threshold, set X to empty
            if (X.shape[0] - len(rmXidx) <= opt.initial_k) or i == opt.iteration_threshold:
                rmCidx = np.arange(k)
                rmXidx = np.arange(X.shape[0])
                X = np.array([])
                y = np.array([])
            # 3.2: Else, remove observations within homogeneous regions from X and reset k to initial_k
            else:
                X = np.delete(X, rmXidx, axis=0)
                y = np.delete(y, rmXidx, axis=0)
                k = opt.initial_k

        # Step 4: If there isn't any homogeneous region
        else:
            # 4.1: Update separability parameter ?????????????????????
            if i > 0 and opt.update_s_parameter and ClassificationError != 0 and ClassificationError_previous != 0:
                s_parameter = s_parameter / np.sqrt(ClassificationError / ClassificationError_previous)

            # 4.2: Evaluate cluster separability
            criterion_met, dm = check_separability(X, y, idx, k, nCL, count_matrix, opt.dm_case, s_parameter, opt.dm_threshold, opt.return_full_dm, opt)

            # 4.3: If there is any separable region and the iteration has not reached the threshold, set k = k + 1
            if criterion_met and i < opt.iteration_threshold:
                k = k + 1

            # 4.4: Else, set X to empty
            else:
                rmCidx = np.arange(k)
                rmXidx = np.arange(X.shape[0])
                X = np.array([])
                y = np.array([])

        # Step 5: Prepare for the next iteration
        # 5.1: Update classification error tracking variables
        sum1 = np.sum(nonDominantClassLabelNumber[rmCidx]) / N
        sum2 = sum2 + sum1
        ClassificationError_previous = ClassificationError

        # 5.2: Store iteration information if required
        if opt.return_full_history or rmCidx.shape[0] > 0:
            store_iteration_info(H, i, C, count_matrix, ObservationsPerCluster,
                         DominantClassLabel, DominantClassLabel_Proportion,
                         rmCidx, s_parameter, dm, ClassificationError, sum1, rmXidx)

    # Final settings
    result['C'] = np.vstack(H['rmC']) # Cluster centroids that have been removed
    result['H'] = H
    result['opt'] = opt
    result['InsampleError'] = np.sum(np.vstack(H['sum1']))
    result['NumIterations'] = i
    result['ElapsedTime'] = time.time() - startTime

    return result
#################################################
#### FSP ALGORITHM BLOCK END
#################################################