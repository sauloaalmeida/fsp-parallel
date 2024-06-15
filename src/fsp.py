from sklearn.cluster import KMeans
from src.divergence_measure.scipy import DivergenceMeasureScipy
from src.options import Options
import numpy as np
import time

class Fsp:

    def __init__(self, X = [], y = [],opt = Options()):
        self.X = X
        self.y = y
        self.opt = opt


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
                'ClassLabel_Proportion': [],
                'DominantClassLabel': [],
                'DominantClassLabel_Proportion': [],
                'ClusterClassificationError': [],
                'ClassificationError': [],
                'sum1': [],
                'rmXidx': [],
                'ElapsedTime': []
            }
        return rmH, H

    def check_separability_criteria(self, k,nCL, dm_threshold, X, ClassLabel_Num, ClassLabel_Indices, opt):
        mat_dm = np.zeros((k,nCL*(nCL-1)//2))
        criterion_met = False
        for c in range(k):
            for a in range(0,nCL-1):
                Na = ClassLabel_Num[c,a]
                if Na < 2: continue
                for b in range(a+1,nCL):
                    Nb = ClassLabel_Num[c,b]
                    if Nb < 2: continue
                    dmAlgorithm = DivergenceMeasureScipy(
                        X[ClassLabel_Indices[c,a], :],
                        X[ClassLabel_Indices[c,b], :]
                    )
                    dm = dmAlgorithm.dm2()
                    mat_dm[c,a*(nCL-1)-a*(a-1)//2+b-a-1] = dm
                    if dm >= dm_threshold:
                        criterion_met = True
                        if not opt.return_full_dm:
                            return criterion_met, mat_dm
        return criterion_met, mat_dm

    def fit(self, X, y):
        self.X = X
        self.y = y
        return self.fit()

    def fit(self):

        startTime = time.time()
        print(f'Tam dp X={self.X.size}, Tam do y={self.y.size}, Options={self.opt}')

        #Initializing history structures
        rmH, H = self.initializeHistoryStructures()

        #Define initial values
        k = self.opt.initial_k       #Current value of k for k-means
        i = -1                   #iteration counter
        rmi = 0                 #The iteration counterfor when cluster removal occured
        nCl = len(np.unique(self.y)) #Number of unique class labels
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

            #initializing structures to calculo de homogeneidade
            Indices = np.zeros(k, dtype=object)
            Indices_Num = np.zeros(k)

            ClassLabel_Indices = np.zeros((k,nCl), dtype=object)
            ClassLabel_Num = np.zeros((k,nCl))
            ClassLabel_Proportion = np.zeros((k,nCl))

            DominantClassLabel = np.zeros(k)
            DominantClassLabel_Proportion = np.zeros(k)
            nonDominantClassLabelNumber = np.zeros(k)

            dm = np.array([])
            rmXidx = np.array([])
            sum1 = np.array([])

            #cluster's evaluation
            for c in range(k):
                #Indices of the observations in X into cluster c
                Indices[c] = np.where(idx == c)[0]
                #Number of observations in X into cluster c
                Indices_Num[c] = len(Indices[c])
                #Class labels of the observations in X into cluster c
                yc = self.y[Indices[c]]

                #for each class label
                for l in range(0,nCl):
                    ClassLabel_Indices[c,l] = np.where(yc == l)[0]
                    ClassLabel_Num[c,l] = len(ClassLabel_Indices[c,l])
                    ClassLabel_Proportion[c,l] = ClassLabel_Num[c,l] / Indices_Num[c]

                # Cluster classification error
                # Let Xc and yc be the observations and class labels in cluster c
                # N the oritional number of observations
                # Nc the number of observations in cluster c

                # Let lc be the dominant class label in cluster c
                lc = np.argmax(ClassLabel_Proportion[c])
                DominantClassLabel[c] = lc

                # Let pc be the proportion of the dominant class label in cluste c
                DominantClassLabel_Proportion[c] = ClassLabel_Proportion[c,lc]

                # The cluster classification error is defined as:
                # ClusterClassificationError(C) = sum(yc != lc) / Nc === mean(yc != lc)
                nonDominantClassLabelNumber[c] = np.sum(yc != c) # Contando os resultados true da verificacao

            #Calcular do erro de cada cluster (local)
            ClusterClassificationError = np.divide(nonDominantClassLabelNumber, Indices_Num)

            # Calculate the global Classification error
            ClassificationError = sum2 + sum(nonDominantClassLabelNumber) / N

            #Verificar criterio de homogeneidade
            vec_max = np.max(ClassLabel_Proportion,1)
            vec = (vec_max >= (1 - self.opt.p_parameter)) & (Indices_Num > self.opt.h_threshold)

            # Se existe cluste p-quase homogeneo
            if np.sum(vec) > 0:
                rmCidx = np.where(vec)[0]
                rmXidx = np.concatenate( Indices[rmCidx] )
                rmC = C[rmCidx, :]
                # rmXidx = rmXidx.astype(int) O Gabriel fez isso, não sabemos o porque

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
                rmH['rmDominantClassLabel'].append(DominantClassLabel[vec])
                rmH['rmDominantClassLabel_Proportion'].append(DominantClassLabel_Proportion[vec])
                rmH['rmXidx'].append(rmXidx)
                rmH['sum1'].append(sum1)

                # rmH_list.append(rmH.copy())  O Gabriel fez isso, não sabemos o porque
                # rmH, _ = initializeStructures(opt)  O Gabriel fez isso, não sabemos o porque

                self.X = np.delete(self.X, rmXidx, axis=0)
                self.y = np.delete(self.y, rmXidx, axis=0)
                k = self.opt.initial_k

            # If there is no homogenous region
            else:
                if i > 1 and self.opt.update_dm_threshold and ClassificationError != 0 and ClassificationError_previous != 0:
                    dm_threshold = dm_threshold_previous / np.sqrt(ClassificationError / ClassificationError_previous)

                #Calculating the divergence measure

                criterion_met, dm = self.check_separability_criteria(k,nCl, dm_threshold, self.X, ClassLabel_Num, ClassLabel_Indices, self.opt)

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
                H['ClassLabel_Proportion'].append(ClassLabel_Proportion)
                H['DominantClassLabel'].append(DominantClassLabel)
                H['DominantClassLabel_Proportion'].append(DominantClassLabel_Proportion)
                H['ClusterClassificationError'].append(ClusterClassificationError)
                H['ClassificationError'].append(ClassificationError)
                H['ClassLabel_Proportion'].append(ClassLabel_Proportion)
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