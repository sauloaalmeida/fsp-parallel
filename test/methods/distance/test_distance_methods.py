import pytest
import numpy as np
from sklearn import datasets
import fsp.methods.distance.scipy_single_thread as dist_scipy_st
import fsp.methods.distance.sklearn_multi_thread as dist_sklearn_mt
import fsp.methods.distance.torch_multi_thread_cpu as dist_torch_mt_cpu
import fsp.methods.distance.torch_gpu as dist_torch_gpu
import fsp.methods.distance.rapidsai_gpu as dist_rapidsai_gpu

class TestDistanceMethods:

    def detalheMat(self, matriz):
        print('\n')

        for linha in matriz:
            for elemento in linha:
                print(elemento,end='\n')

        print('\n')

    def detalheVet(self, vetor):
        print('\n')

        for elemento in vetor:
            print(elemento,end='\n')

        print('\n')

    def setup_method(self, method):
        np.random.seed(42)
        observations=1000
        features=100
        epsilon = 1e-10
        self.a = np.random.rand(observations,features)
        self.b = np.random.rand(observations,features)
        self.va = np.var(self.a, axis=0)
        self.vb = np.var(self.b, axis=0)
        self.ha2 = np.random.uniform(epsilon, 1 - epsilon)
        self.hb2 = np.random.uniform(epsilon, 1 - epsilon)

    def test_scipy_eq_sklearn_pdist_dm1(self):
        assert np.array_equal(dist_scipy_st.pdist_dm1(self.a),dist_sklearn_mt.pdist_dm1(self.a))

    # def test_scipy_eq_sklearn_pdist_dm2(self):
    #     assert np.array_equal(dist_scipy_st.pdist_dm2(self.a, self.va), dist_sklearn_mt.pdist_dm2(self.a, self.va))

    def test_scipy_eq_sklearn_cdist(self):
        assert np.allclose(dist_scipy_st.cdist(self.a, self.b), dist_sklearn_mt.cdist(self.a, self.b))

    def test_scipy_eq_sklearn_cdist_dm1(self):
        assert np.array_equal(dist_scipy_st.cdist_dm1(self.a, self.b), dist_sklearn_mt.cdist_dm1(self.a, self.b))

    # def test_scipy_eq_sklearn_cdist_dm2(self):
    #     assert np.array_equal(dist_scipy_st.cdist_dm2(self.a, self.b, self.va, self.vb, self.ha2, self.hb2), dist_sklearn_mt.cdist_dm2(self.a, self.b, self.va, self.vb, self.ha2, self.hb2))

    def test_scipy_eq_torch_cdist_cpu(self):
        assert np.allclose(dist_scipy_st.cdist(self.a, self.b), dist_torch_mt_cpu.cdist(self.a, self.b))

    def test_scipy_eq_torch_cdist_gpu(self):
        assert np.allclose(dist_scipy_st.cdist(self.a, self.b), dist_torch_gpu.cdist(self.a, self.b))

    def test_scipy_eq_torch_cdist_dm1_cpu(self):
        assert np.allclose(dist_scipy_st.cdist_dm1(self.a, self.b), dist_torch_mt_cpu.cdist_dm1(self.a, self.b))

    def test_scipy_eq_torch_cdist_dm1_gpu(self):
        assert np.allclose(dist_scipy_st.cdist_dm1(self.a, self.b), dist_torch_gpu.cdist_dm1(self.a, self.b))

    def test_scipy_eq_torch_pdist_dm1_cpu(self):
        assert np.allclose(dist_scipy_st.pdist_dm1(self.a), dist_torch_mt_cpu.pdist_dm1(self.a))

    def test_scipy_eq_torch_pdist_dm1_gpu(self):
        assert np.allclose(dist_scipy_st.pdist_dm1(self.a), dist_torch_gpu.pdist_dm1(self.a))

    # def test_scipy_eq_torch_pdist_dm2_cpu(self):
    #     assert np.allclose(dist_scipy_st.pdist_dm2(self.a, self.va), dist_torch_mt_cpu.pdist_dm2(self.a, self.va))

    def test_scipy_eq_rapidsai_cdist_gpu(self):
        assert np.allclose(dist_scipy_st.cdist(self.a, self.b), dist_rapidsai_gpu.cdist(self.a, self.b))

    def test_scipy_eq_rapidsai_pdist_dm1_cpu(self):
        assert np.allclose(dist_scipy_st.pdist_dm1(self.a), dist_rapidsai_gpu.pdist_dm1(self.a))

    def test_scipy_eq_rapidsai_cdist_dm1_gpu(self):
        assert np.allclose(dist_scipy_st.cdist_dm1(self.a, self.b), dist_rapidsai_gpu.cdist_dm1(self.a, self.b))
