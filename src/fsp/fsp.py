from sklearn.cluster import KMeans
from fsp.divergence_measure.scipy import DivergenceMeasureScipy
from fsp.options import Options
from scipy.spatial import distance
import numpy as np
import time

class Fsp:

    def __init__(self, opt = Options()):
        self.X = []
        self.y = []
        self.opt = opt

    def predict_case1(self, Xtest,rmH,Xtrain,ytrain):
        # Step 0: Retrieve the cluster centroid removed by the FSP method
        C = np.vstack(rmH['rmC']) # k x d matrix of cluster centroids

        # Step 1: Classify the Xtrain dataset by finding the nearest centroid for each observation
        dist_Xtrain_C = distance.cdist(Xtrain, C, 'euclidean')  # Ntrain x k matrix of distances between Xtrain observations and centroids
        Itrain = np.argmin(dist_Xtrain_C, axis=1)  # Array of size Ntrain, index of the nearest centroid for each observation in Xtrain

        # Step 2: Calculate the dominant class label and its proportion for each cluster in the training dataset
        k = C.shape[0]  # Number of clusters
        n = np.unique(ytrain).size  # Number of unique classes
        count_matrix = np.zeros((k, n), dtype=int)  # Initialize a k x n matrix to count occurrences of each class in each cluster
        np.add.at(count_matrix, (Itrain, ytrain), 1)  # Count the occurrences of each class in each cluster

        DominantClassLabel = np.argmax(count_matrix, axis=1)  # Dominant class label for each cluster
        DominantClassLabel_Proportion = count_matrix[np.arange(k), DominantClassLabel] / np.sum(count_matrix, axis=1)  # Proportion of the dominant class in each cluster
        #DominantClassLabel_Proportion = np.max(count_matrix, axis=1) / np.sum(count_matrix, axis=1)

        # Step 3: Classify the Xtest dataset by finding the nearest centroid for each observation
        dist_Xtest_C = distance.cdist(Xtest, C, 'euclidean')  # Ntest x k matrix of distances between Xtest points and centroids
        Itest = np.argmin(dist_Xtest_C, axis=1)  # Array of size Ntest, index of the nearest centroid for each observation in Xtest

        # Step 4: Assign the predicted class label and its proportion for each observation in Xtest
        ypredict = DominantClassLabel[Itest] # Predicted class labels for Xtest
        ypredict_Proportion = DominantClassLabel_Proportion[Itest] # Proportion of the predicted class labels for Xtest

        return ypredict, ypredict_Proportion


    def check_homogeneity(self, y, idx, k:int, nCL:int, p_parameter:float, h_threshold:int) -> tuple:
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
        tuple: (rmCidx, rmXidx, DominantClassLabel, DominantClassLabel_Proportion, ObservationsPerCluster, nonDominantClassLabelNumber)
            - p_almost_homogeneous_clusters (numpy.ndarray): Indices of clusters that satisfy the homogeneity criterion.
            - rmXidx (numpy.ndarray): Indices of observations in clusters that satisfy the homogeneity criterion.
            - DominantClassLabel (numpy.ndarray): k-element vector with the dominant class label for each cluster.
            - DominantClassLabel_Proportion (numpy.ndarray): k-element vector with the proportion of the dominant class in each cluster.
            - ObservationsPerCluster (numpy.ndarray): k-element vector with the number of observations in each cluster.
            - nonDominantClassLabelNumber (numpy.ndarray): k-element vector with the number of observations that do not belong to the dominant class in each cluster.
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
        rmCidx = np.where( (DominantClassLabel_Proportion >= (1 - p_parameter)) & (ObservationsPerCluster >= h_threshold) )[0]

        # Step 4: Determine which observations are in clusters that satisfy the homogeneity criterion
        #rmXidx = np.flatnonzero(np.isin(idx, rmCidx))
        rmXidx = np.where(np.isin(idx, rmCidx))[0]

        return rmCidx, rmXidx, DominantClassLabel, DominantClassLabel_Proportion, ObservationsPerCluster, nonDominantClassLabelNumber, count_matrix


    def check_separability(self,X, y, idx, k:int, nCL:int, dm_threshold:float, return_full_dm:bool, dm_case:int=2):
        """
        Check the separability of clusters based on divergence measures.

        Parameters:
        X (numpy.ndarray): m x d matrix where each row contains the d-dimensional coordinates of an observation (m is the number of observations, d is the number of features).
        y (numpy.ndarray): m-element vector where each entry contains the label of the corresponding observation. Labels are integers between 0 and nCL-1 (nCL is the number of unique class labels).
        idx (numpy.ndarray): m-element vector where each entry contains the index of the nearest centroid for the corresponding observation, as determined by the k-means method.
        k (int): Number of clusters.
        nCL (int): Number of unique class labels.
        dm_threshold (float): Threshold value for divergence measure.
        return_full_dm (bool): If True, the full divergence measure matrix is calculated. Otherwise, return when the separability criterion is met.
        dm_case (int, optional): Divergence measure case (default is 2).

        Returns:
        tuple: (criterion_met, mat_dm)
            - criterion_met (bool): True if the separability criterion is met, False otherwise.
            - mat_dm (numpy.ndarray): k x (nCL - 1) matrix of divergence measures.
        """

        criterion_met = False
        mat_dm = np.zeros((k,nCL*(nCL-1)//2))

        for c in range(k):
            idx_c = np.where(idx == c)[0] # Indices of observations in cluster c
            X_c = X[idx_c,:] # Observations in cluster c
            y_c = y[idx_c] # Class labels of observations in cluster c
            for a in range(0,nCL-1):
                idx_c_a = np.where(y_c == a)[0] # Indices of observations in X_c with class a
                Na = len(idx_c_a) # Number of observations in X_c with class a
                if Na < 2: continue
                X_c_a = X_c[idx_c_a,:]
                for b in range(a+1,nCL):
                    idx_c_b = np.where(y_c == b)[0] # Indices of observations in X_c with class b
                    Nb = len(idx_c_b) # Number of observations in X_c with class b
                    if Nb < 2: continue

                    dmAlgorithm = DivergenceMeasureScipy(# Calculate divergence measure
                        X_c_a,
                        X_c[idx_c_b,:]
                    )
                    dm = dmAlgorithm.dm2()
                    mat_dm[c,a*(nCL-1)-a*(a-1)//2+b-a-1] = dm
                    if dm >= dm_threshold:
                        criterion_met = True
                        if not return_full_dm:
                            return criterion_met, mat_dm
        return criterion_met, mat_dm


    def initializeHistoryStructures(self):
        rmH = {
            'i': [],
            'k': [],
            'C': [],
            'rmCidx': [],
            'rmC': [],
            'dm_threshold': [],
            'dm': [],
            'rmDominantClassLabel': [],
            'rmDominantClassLabel_Proportion': [],
            'sum1': [],
            'rmXidx': []
        }
        H = {}

        if (self.opt.return_full_history):
            H = {
                'i': [],
                'k': [],
                'C': [],
                'dm_threshold': [],
                'dm': [],
                'count_matrix': [],
                'DominantClassLabel': [],
                'DominantClassLabel_Proportion': [],
                'ClusterClassificationError': [],
                'ClassificationError': [],
                'sum1': [],
                'rmXidx': [],
                'ElapsedTime': []
            }
        return rmH, H

    def fit(self, X, y):
        startTime = time.time()

        #initializing data
        self.X = X
        self.y = y

        #Initializing history structures
        rmH, H = self.initializeHistoryStructures()

        #Define initial values
        k = self.opt.initial_k       #Current value of k for k-means
        i = -1                   #iteration counter
        rmi = 0                 #The iteration counterfor when cluster removal occured
        nCL = len(np.unique(self.y)) #Number of unique class labels
        N = self.X.shape[0]          #Number of observations
        p_parameter = self.opt.p_parameter #The p-almost homogeneous parameter
        dm_threshold = self.opt.dm_threshold #The divergence measure threshold
        sum2 = 0                #This variable is employed in the calculation of the classification


        while self.X.shape[0] > 0:
            i += 1 #contador da iteracao

            if (self.opt.return_full_history):
                tStart_while = time.time()
                H['i'].append(i)
                H['k'].append(k)

            #Running kmeans
            kmeans = KMeans(n_clusters=k,max_iter= 1000, random_state=self.opt.kmeans_random_state,  n_init=1).fit(self.X)
            idx = kmeans.labels_
            C = kmeans.cluster_centers_

            #Verificar criterio de homogeneidade
            (rmCidx,
            rmXidx,
            DominantClassLabel,
            DominantClassLabel_Proportion,
            ObservationsPerCluster,
            nonDominantClassLabelNumber,
            count_matrix) = self.check_homogeneity(self.y, idx, k, nCL, self.opt.p_parameter, self.opt.h_threshold)

            #Calcular do erro de cada cluster (local)
            ClusterClassificationError = np.divide(nonDominantClassLabelNumber, ObservationsPerCluster)

            # Calculate the global Classification error
            ClassificationError = sum2 + sum(nonDominantClassLabelNumber) / N

            # Se existe cluste p-quase homogeneo
            sum1 = np.array([])
            dm = np.array([])
            if len(rmCidx) > 0:
                rmC = C[rmCidx, :]

                rmNonDominantClassLabelNumberbyN = nonDominantClassLabelNumber[rmCidx] / N
                sum1 = sum(rmNonDominantClassLabelNumberbyN)
                sum2 = sum2 + sum1

                rmH['i'].append(i)
                rmH['k'].append(k)
                rmH['C'].append(C)
                rmH['rmCidx'].append(rmCidx)
                rmH['rmC'].append(rmC)
                rmH['dm_threshold'].append(dm_threshold)
                rmH['dm'].append(dm)
                rmH['rmDominantClassLabel'].append(DominantClassLabel[rmCidx])
                rmH['rmDominantClassLabel_Proportion'].append(DominantClassLabel_Proportion[rmCidx])
                rmH['rmXidx'].append(rmXidx)
                rmH['sum1'].append(sum1)

                self.X = np.delete(self.X, rmXidx, axis=0)
                self.y = np.delete(self.y, rmXidx, axis=0)
                k = self.opt.initial_k

            # If there is no homogenous region
            else:
                if i > 1 and self.opt.update_dm_threshold and ClassificationError != 0 and ClassificationError_previous != 0:
                    dm_threshold = dm_threshold_previous / np.sqrt(ClassificationError / ClassificationError_previous)

                #Calculating the divergence measure
                criterion_met, dm = self.check_separability(self.X, self.y, idx, k, nCL, dm_threshold, self.opt.return_full_dm, self.opt.dm_case);

                # YES, there is separable region. Set k = k + 1.
                if criterion_met:
                    k = k + 1
                # NO, there isn't any separable region. Set X to empty.
                else:
                    rmNonDominantClassLabelNumberbyN = nonDominantClassLabelNumber / N
                    sum1 = np.sum(rmNonDominantClassLabelNumberbyN)
                    sum2 = sum2 + sum1

                    rmCidx = np.arange(k)
                    rmC = C
                    rmXidx = np.arange(self.X.shape[0])

                    rmH['i'].append(i)
                    rmH['k'].append(k)
                    rmH['C'].append(C)
                    rmH['rmCidx'].append(rmCidx)
                    rmH['rmC'].append(rmC)
                    rmH['dm_threshold'].append(dm_threshold)
                    rmH['dm'].append(dm)
                    rmH['rmDominantClassLabel'].append(DominantClassLabel)
                    rmH['rmDominantClassLabel_Proportion'].append(DominantClassLabel_Proportion)
                    rmH['rmXidx'].append(rmXidx)
                    rmH['sum1'].append(sum1)

                    self.X = np.array([])
                    self.y = np.array([])

            # Final settings
            dm_threshold_previous = dm_threshold
            ClassificationError_previous = ClassificationError

            # Fill full history
            if self.opt.return_full_history:
                H['dm_threshold'].append(dm_threshold)
                H['dm'].append(dm)
                H['count_matrix'].append(count_matrix)
                H['DominantClassLabel'].append(DominantClassLabel)
                H['DominantClassLabel_Proportion'].append(DominantClassLabel_Proportion)
                H['ClusterClassificationError'].append(ClusterClassificationError)
                H['ClassificationError'].append(ClassificationError)
                H['sum1'].append(sum1)
                H['C'].append(C)
                H['rmXidx'].append(rmXidx)
                H['ElapsedTime'].append(time.time() - tStart_while)


            # Verificar limitação da iteração while
            if i >= self.opt.iteration_threshold:
                print('WARNING: the number of interactions reached the threshold. There are still observations in X, I need to rethink this case.')
                break

        # Fill the FSP output
        result = {'rmH':rmH,'H':H,'opt':self.opt,'InSampleError':np.sum(rmH['sum1']),'runtime':time.time() - startTime}
        return result


    def helloFsp(self):
        return "hello fsp"