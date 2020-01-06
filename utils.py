from sklearn.preprocessing import normalize
import numpy as np
import pickle as pkl
import networkx as nx
from scipy.special import iv
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
from scipy.optimize import minimize
from scipy.integrate import quad
import sys
import math
import time

def laplacian(W):
	"""Return the Laplacian of the weight matrix."""
	# Degree matrix.
	d = W.sum(axis=0)
	# Laplacian matrix.
	d = 1 / np.sqrt(d)
	D = sp.diags(d.A.squeeze(), 0)
	I = sp.identity(d.size, dtype=W.dtype)
	L = I - D * W * D

	assert type(L) is sp.csr.csr_matrix
	return L

def weight_wavelet(s,lamb,U):
	s = s
	for i in range(len(lamb)):
		lamb[i] = math.exp(-lamb[i]*s)

	Weight = np.dot(np.dot(U, np.diag(lamb)),np.transpose(U))

	return Weight

def weight_wavelet_inverse(s,lamb,U):
	s = s
	for i in range(len(lamb)):
		lamb[i] = math.exp(lamb[i] * s)

	Weight = np.dot(np.dot(U, np.diag(lamb)), np.transpose(U))

	return Weight

def fourier(L, algo='eigh', k=100):
	"""Return the Fourier basis, i.e. the EVD of the Laplacian."""
	
	def sort(lamb, U):
		idx = lamb.argsort()
		return lamb[idx], U[:, idx]

	if algo is 'eig':
		lamb, U = np.linalg.eig(L.toarray())
		lamb, U = sort(lamb, U)
	elif algo is 'eigh':
		lamb, U = np.linalg.eigh(L.toarray())
		lamb, U = sort(lamb, U)
	elif algo is 'eigs':
		lamb, U = sp.linalg.eigs(L, k=k, which='SM')
		lamb, U = sort(lamb, U)
	elif algo is 'eigsh':
		lamb, U = sp.linalg.eigsh(L, k=k, which='SM')

	return lamb, U

def wavelet_basis(adj,s,threshold):

	L = laplacian(adj)

	print('Eigendecomposition start...')
	start = time.time()

	lamb, U = fourier(L)

	elapsed = (time.time() - start)
	print(f'Eigendecomposition complete, Time used: {elapsed:.6g}s')

	print('Calculating wavelet...')
	start = time.time()

	Weight = weight_wavelet(s,lamb,U)
	inverse_Weight = weight_wavelet_inverse(s,lamb,U)

	elapsed = (time.time() - start)
	print(f'Wavelet get, Time used: {elapsed:.6g}s')
	del U,lamb

	print('Threshold to zero...')
	start = time.time()

	Weight[Weight < threshold] = 0.0
	inverse_Weight[inverse_Weight < threshold] = 0.0

	elapsed = (time.time() - start)
	print(f'Threshold complete, Time used: {elapsed:.6g}s')

	print('L1 normalizing...')
	start = time.time()

	Weight = normalize(Weight, norm='l1', axis=1)
	inverse_Weight = normalize(inverse_Weight, norm='l1', axis=1)

	elapsed = (time.time() - start)
	print(f'L1 normalizing complete, Time used: {elapsed:.6g}s')

	Weight = sp.coo_matrix(Weight)
	inverse_Weight = sp.coo_matrix(inverse_Weight)

	t_k = (Weight, inverse_Weight)
	return t_k

def largest_lamb(L):
	lamb, U = sp.linalg.eigsh(L, k=1, which='LM')
	lamb = lamb[0]
	#print(lamb)
	return lamb

def fast_wavelet_basis(adj,s,threshold,m):
	L = laplacian(adj)
	lamb = largest_lamb(L)

	print('Calculating wavelet...')
	start = time.time()
	a = lamb / 2
	c = []
	inverse_c = []
	for i in range(m + 1):
		f = lambda x: np.cos(i * x) * np.exp(s * a * (np.cos(x) + 1))
		inverse_f = lambda x: np.cos(i * x) * np.exp(-s * a * (np.cos(x) + 1))

		f_res = 2 * np.exp(s * a) * iv(i, s * a)
		inverse_f_res = 2 * np.exp(-s * a) * iv(i, -s * a)
		
		# Compare with result of numerical computation
		print(f'Difference in order {i}: ')
		print(f'{f_res - quad(f, 0, np.pi)[0] * 2 / np.pi:.3g}')
		print(f'{inverse_f_res - quad(inverse_f, 0, np.pi)[0] * 2 / np.pi:.3g}')

		c.append(f_res)
		inverse_c.append(inverse_f_res)

	T = [sp.eye(adj.shape[0])]
	T.append((1. / a) * L - sp.eye(adj.shape[0]))

	temp = (2. / a) * (L - sp.eye(adj.shape[0]))

	for i in range(2, m + 1):
		T.append(temp.dot(T[i - 1]) - T[i - 2])

	Weight = c[0] / 2 * sp.eye(adj.shape[0])
	inverse_Weight = inverse_c[0] / 2 * sp.eye(adj.shape[0])

	for i in range(1, m + 1):
		Weight += c[i] * T[i]
		inverse_Weight += inverse_c[i] * T[i]

	elapsed = (time.time() - start)
	print(f'Wavelet get, Time used: {elapsed:.6g}s')

	Weight, inverse_Weight = Weight.tocoo(), inverse_Weight.tocoo()

	#print((Weight.dot(inverse_Weight)).toarray())
	print('Threshold to zero...')
	start = time.time()

	Weight = threshold_to_zero(Weight, threshold)
	inverse_Weight = threshold_to_zero(inverse_Weight, threshold)

	elapsed = (time.time() - start)
	print(f'Threshold complete, Time used: {elapsed:.6g}s')

	print('L1 normalizing...')
	start = time.time()

	Weight = normalize(Weight, norm='l1', axis=1)
	inverse_Weight = normalize(inverse_Weight, norm='l1', axis=1)

	elapsed = (time.time() - start)
	print(f'L1 normalizing complete, Time used: {elapsed:.6g}s')

	t_k = (Weight, inverse_Weight)
	return t_k

def threshold_to_zero(mx, threshold):
    """Set value in a sparse matrix lower than
     threshold to zero. 
    
    Return the 'coo' format sparse matrix.

    Parameters
    ----------
    mx : array_like
        Sparse matrix.
    threshold : float
        Threshold parameter.
    """
    high_values_indexes = set(zip(*((np.abs(mx) >= threshold).nonzero())))
    nonzero_indexes = zip(*(mx.nonzero()))

    if not sp.isspmatrix_lil(mx):
        mx = mx.tolil()   

    for s in nonzero_indexes:
        if s not in high_values_indexes:
            mx[s] = 0.0
    mx = mx.tocoo()
    mx.eliminate_zeros()
    return mx
	


