
import torch
import numpy as np
from scipy.spatial import distance

class TestDmTorch:

    def setup_method(self, method):
        np.random.seed(42)
        observations=100
        features=10
        self.A = np.random.rand(observations,features)
        self.B = np.random.rand(observations,features)
        self.Va = np.var(self.A, axis=0)
        self.Vb = np.var(self.B, axis=0)
        self.tensorA = torch.tensor(self.A, dtype=torch.float64).to("cpu")
        self.tensorB = torch.tensor(self.B, dtype=torch.float64).to("cpu")
        self.tVa = torch.var(self.tensorA, dim=0, correction=0)
        self.tVb = torch.var(self.tensorB, dim=0, correction=0)

    def _getShapes(self):
        #Define Na and Nb, d
        Na, d = self.A.shape
        Nb, _ = self.B.shape

        return Na, d, Nb

    def _calculateConstValuesDmCase1(self):

        #get shapes of datasets
        Na, d, Nb = self._getShapes()

        #define values not vectorized (consts for this execution)
        ha2 = ( 4/((2*d+1)*Na) )**(2/(d+4))
        hb2 = ( 4/((2*d+1)*Nb) )**(2/(d+4))

        logha2 = (2/(d+4))*( np.log(4) - np.log(2*d+1) - np.log(Na) )
        loghb2 = (2/(d+4))*( np.log(4) - np.log(2*d+1) - np.log(Nb) )

        return Na, d, Nb, ha2, hb2, logha2, loghb2

    def _calculateConstValuesDmCase2(self):

        #get shapes of datasets
        Na, d, Nb = self._getShapes()

        #define values not vectorized (consts for this execution)
        ha2 = ( 4/((d+2)*Na) )**(2/(d+4))
        hb2 = ( 4/((d+2)*Nb) )**(2/(d+4))

        logha2 = (2/(d+4))*( np.log(4) - np.log(d+2) - np.log(Na) )
        loghb2 = (2/(d+4))*( np.log(4) - np.log(d+2) - np.log(Nb) )

        return Na, d, Nb, ha2, hb2, logha2, loghb2

    def test_variance(self):

        #compare results
        assert np.allclose(self.Va,self.tVa.numpy())

    def test_calc_logha2(self):
        Na, d = self.A.shape

        logha2 = (2/(d+4))*( np.log(4) - np.log(2*d+1) - np.log(Na) )

        tLogha2 = (2/(d+4))*( torch.log(torch.tensor(4, dtype=torch.float64).to("cpu")) - torch.log(torch.tensor((2*d+1), dtype=torch.float64).to("cpu")) - torch.log(torch.tensor(Na, dtype=torch.float64).to("cpu")) )

        assert np.allclose(logha2,tLogha2.numpy())

    def test_calc_meanVa(self):

        meanVa = np.mean(self.Va)

        tMeanVa = torch.mean(self.tVa)

        assert np.allclose(meanVa,tMeanVa.numpy())

    def test_calc_logprodVa1(self):

        logprodVa = np.sum(np.log(self.Va))
        tLogprodVa = torch.sum(torch.log(self.tVa))

        assert np.allclose(logprodVa,tLogprodVa.numpy())

    def test_calc_logprodVa2(self):
        Na, d = self.A.shape
        Nb, _ = self.B.shape

        ha2 = ( 4/((d+2)*Na) )**(2/(d+4))
        hb2 = ( 4/((d+2)*Nb) )**(2/(d+4))

        logprodVab = np.sum(np.log(ha2*self.Va+hb2*self.Vb))

        tHa2 = torch.tensor(ha2, dtype=torch.float64).to("cpu")
        tHb2 = torch.tensor(hb2, dtype=torch.float64).to("cpu")

        tLogprodVab = torch.sum(torch.log(tHa2*self.tVa+tHb2*self.tVb))

        assert np.allclose(logprodVab,tLogprodVab.numpy())

    def test_calc_logprodVa3(self):
        Na, d = self.A.shape
        Nb, _ = self.B.shape

        ha2 = ( 4/((d+2)*Na) )**(2/(d+4))
        hb2 = ( 4/((d+2)*Nb) )**(2/(d+4))

        logprodVab = np.sum(np.log(ha2*self.Va+hb2*self.Vb))
        tLogprodVab = torch.sum(torch.log(ha2*self.tVa+hb2*self.tVb))

        assert np.allclose(logprodVab,tLogprodVab.numpy())

    def test_calc_sqrt_weight(self):
        Na, d = self.A.shape
        Nb, _ = self.B.shape

        ha2 = ( 4/((d+2)*Na) )**(2/(d+4))
        hb2 = ( 4/((d+2)*Nb) )**(2/(d+4))

        w = np.sqrt(ha2*self.Va+hb2*self.Vb)

        tw = torch.sqrt(ha2*self.tVa+hb2*self.tVb)

        assert np.allclose(w,tw.numpy())

    def test_calc_pdist_a(self):
        Na, d = self.A.shape
        ha2 = ( 4/((d+2)*Na) )**(2/(d+4))
        meanVa = np.mean(self.Va)
        pdist  = np.sum(np.exp( -distance.pdist(self.A,'sqeuclidean')/(4*ha2*meanVa)))
        tpdist = torch.sum(torch.exp( -torch.pow(torch.nn.functional.pdist(self.tensorA),2)/(4*ha2*meanVa)))

        assert np.allclose(pdist,tpdist.numpy())

    def test_compare_dm_case1_sumab_zero(self):

        zeros = np.zeros((3,5))
        tZeros = torch.tensor(zeros, dtype=torch.float64).to("cpu")
        tD_CS = torch.sum(tZeros)

        assert 0 == tD_CS

    def test_dm_case1_sumab_diff_zero(self):
        Na, d, Nb, ha2, hb2, logha2, loghb2 = self._calculateConstValuesDmCase1()

        tMeanVa = torch.mean(self.tVa)
        tMeanVb = torch.mean(self.tVb)
        meanVa = np.mean(self.Va)
        meanVb = np.mean(self.Vb)

        tSum_ab = torch.sum(torch.exp( -torch.cdist(self.tensorA, self.tensorB)/(2*ha2*tMeanVa+2*hb2*tMeanVb)))
        sum_ab = np.sum(np.exp( -distance.cdist(self.A, self.B)/(2*ha2*meanVa+2*hb2*meanVb) ))

        tSum_a = torch.sum(torch.exp( -torch.pow(torch.nn.functional.pdist(self.tensorA),2)/(4*ha2*tMeanVa)))
        tSum_b = torch.sum(torch.exp( -torch.pow(torch.nn.functional.pdist(self.tensorB),2)/(4*hb2*tMeanVb)))

        sum_a  = np.sum(np.exp( -distance.pdist(self.A,'sqeuclidean')/(4*ha2*meanVa)))
        sum_b  = np.sum(np.exp( -distance.pdist(self.B,'sqeuclidean')/(4*hb2*meanVb)))

        tD_CS = -d*torch.log(torch.tensor(2, dtype=torch.float64).to("cpu")) - (d/2)*( logha2 + loghb2 + torch.log(tMeanVa) + torch.log(tMeanVb) ) + d*torch.log(ha2*tMeanVa+hb2*tMeanVb) - 2*torch.log(tSum_ab) +  torch.log(Na + 2*tSum_a) +  torch.log(Nb + 2*tSum_b)
        D_CS = -d*np.log(2) - (d/2)*( logha2 + loghb2 + np.log(meanVa) + np.log(meanVb) ) + d*np.log(ha2*meanVa+hb2*meanVb) - 2*np.log(sum_ab) +  np.log(Na + 2*sum_a) +  np.log(Nb + 2*sum_b)

        assert np.allclose(D_CS,tD_CS)

    def test_dm_case2_sumab_diff_zero(self):
        Na, d, Nb, ha2, hb2, logha2, loghb2 = self._calculateConstValuesDmCase2()

        self.tVa[self.tVa<10**(-12)] = 10**(-12)
        self.tVb[self.tVb<10**(-12)] = 10**(-12)

        self.Va[self.Va<10**(-12)] = 10**(-12)
        self.Vb[self.Vb<10**(-12)] = 10**(-12)

        assert np.allclose(self.Va,self.tVa.numpy())

        tW = torch.sqrt(ha2*self.tVa+hb2*self.tVb)
        tSum_ab = torch.sum(torch.exp( -torch.pow(torch.cdist(self.tensorA/tW, self.tensorB/tW),2))**2/2)

        w = np.sqrt(ha2*self.Va+hb2*self.Vb)
        sum_ab = np.sum(np.exp( -distance.cdist(self.A/w, self.B/w, metric='sqeuclidean'))**2/2)

        assert np.allclose(w,tW.numpy())
        assert np.allclose(sum_ab,tSum_ab.numpy())

        logprodVa = np.sum(np.log(self.Va))
        logprodVb = np.sum(np.log(self.Vb))
        logprodVab = np.sum(np.log(ha2*self.Va+hb2*self.Vb))

        tLogprodVa = torch.sum(torch.log(self.tVa))
        tLogprodVb = torch.sum(torch.log(self.tVb))
        tLogprodVab = torch.sum(torch.log(ha2*self.tVa+hb2*self.tVb))

        assert np.allclose(logprodVa,tLogprodVa.numpy())
        assert np.allclose(logprodVb,tLogprodVb.numpy())
        assert np.allclose(logprodVab,tLogprodVab.numpy())

        sum_a = np.sum(np.exp( - distance.pdist(self.A/np.sqrt(self.Va), metric='sqeuclidean')**2/(4*ha2)))
        sum_b = np.sum(np.exp( - distance.pdist(self.B/np.sqrt(self.Vb), metric='sqeuclidean')**2/(4*hb2)))

        tSum_a = torch.sum(torch.exp( -torch.pow(torch.nn.functional.pdist(self.tensorA/torch.sqrt(self.tVa)),2)**2/(4*ha2)))
        tSum_b = torch.sum(torch.exp( -torch.pow(torch.nn.functional.pdist(self.tensorB/torch.sqrt(self.tVb)),2)**2/(4*hb2)))

        assert np.allclose(sum_a,tSum_a.numpy())
        assert np.allclose(sum_b,tSum_b.numpy())

        D_CS = -d*np.log(2) - (d/2)*( logha2 + loghb2 ) - (1/2)*( logprodVa + logprodVb ) + logprodVab -2*np.log(sum_ab) + np.log(Na + 2*sum_a) + np.log(Nb + 2*sum_b)
        tD_CS = -d*torch.log(torch.tensor(2, dtype=torch.float64).to("cpu")) - (d/2)*( logha2 + loghb2 ) - (1/2)*( tLogprodVa + tLogprodVb ) + tLogprodVab -2*torch.log(tSum_ab) + torch.log(Na + 2*tSum_a) + torch.log(Nb + 2*tSum_b)

        assert np.allclose(D_CS,tD_CS.numpy())

    def test_filter_update_tensor(self):

        tFiltered = torch.tensor([0.1, -1.56, 10**(-13), 10**(-15), 10**(-11), 1.2, 0])
        tFiltered[tFiltered < 10**(-12)] = 10**(-12)

        tExpected = torch.tensor([0.1, 10**(-12), 10**(-12), 10**(-12), 10**(-11), 1.2, 10**(-12)])

        assert np.allclose(tExpected.numpy(), tFiltered.numpy())