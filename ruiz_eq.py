import numpy as np
import scipy
import re
from util_opt import nnz_codebook

bin_num_dict={'a':8, 'b':4, 'c':2}

def bin_votes(row_char, nnz_col_indices, isca_c):
	bin_num = bin_num_dict.get(row_char)
	bin_width = isca_c//bin_num
	preferred_bin = (nnz_col_indices//bin_width) % bin_num
	""" remove bank collision when computing votes, each bin have many banks"""
	preferred_banks = nnz_col_indices % bin_width
	votes = np.zeros((bin_num), dtype=np.uint16)
	votes_prefix = np.bincount(preferred_bin)
	votes[:len(votes_prefix)]=votes_prefix
	""" remove conflict in votes"""
	for item in range(bin_num):
		bank_access_in_same_bin = preferred_banks[np.where(preferred_bin==item)]
		bank_access_hist = np.bincount(bank_access_in_same_bin)
		bank_access_conflict = bank_access_hist[np.where(bank_access_hist>1)]
		conflict_num = np.sum(bank_access_conflict)-len(bank_access_conflict)
		votes[item] -= conflict_num
	""" break max tie using random choice"""
	top_vote_bin = np.random.choice(np.flatnonzero(votes == votes.max()))
	return top_vote_bin

def row_schedule(bin_queues):
	""" Rescheudle rows based on their preferred bins
	return aligned row indices and some left-over """
	num_of_queues = len(bin_queues)	
	total_rows = sum(map(lambda x:len(x), bin_queues))
	""" the idea is to reduce high bins and fill low bins"""
	average_height = total_rows//num_of_queues
	left_over_rows =[]
	""" Phase 1: Reduce High Bins """
	for q in bin_queues:
		if len(q)>average_height:
			for _ in range(len(q) - average_height):
				left_over_rows.append(q.pop(0))

	""" Phase 2: Fill Low Bins """
	for q in bin_queues:
		if len(q)<average_height:
			for _ in range(average_height -len(q)):
				q.append(left_over_rows.pop(0))

	""" Phase 3: Export aligned Bins"""
	aligned_rows = []
	for _ in range(average_height):
		aligned_rows.extend( q.pop(0) for q in bin_queues)
	
	return aligned_rows, left_over_rows

def struct_adapt(mat_csr, isca_c):
	""" search for a, b, c occurance and rearrange these rows"""
	nnz_char_map = nnz_codebook(isca_c)	

	align_queue={}
	for item in ['a', 'b', 'c']:
		align_queue[item] = [ [] for _ in range(bin_num_dict.get(item))]
	left_over_rows = []

	for row_idx in range(mat_csr.shape[0]):
		row_start = mat_csr.indptr[row_idx]
		row_end = mat_csr.indptr[row_idx+1]
		nnz_col_indices = mat_csr.indices[row_start:row_end]
		nnz_col_cnt = len(nnz_col_indices)

		"""look for a, b, c, ignore e, d """
		if nnz_col_cnt > isca_c//2:
			left_over_rows.append(row_idx)
			continue

		row_char = nnz_char_map.get(nnz_col_cnt)
		assert(row_char is not None)
		bin_idx = bin_votes(row_char, nnz_col_indices, isca_c)
		align_queue[row_char][bin_idx].append(row_idx)

	""" reschedule the rows based on votes,
	using similar ideas in column align"""
	reschduled_rows = []
	
	for item in ['a', 'b', 'c']:
		sr, lr = row_schedule(align_queue[item])
		reschduled_rows.extend(sr)
		left_over_rows.extend(lr)

	reschduled_rows.extend(left_over_rows)	
	assert np.all(np.bincount(reschduled_rows) == 1) and len(reschduled_rows) == mat_csr.shape[0]

	""" Next step, assign the bank based on 
	    if bank is takens, can find nearby positions """
	return reschduled_rows

def set_scalar_if_lt(v, th=1e-4, clip_value=1.0):
	v[v<th]=clip_value

def set_scalar_if_gt(v, th=1e4, clip_value=1e4):
	v[v>th]=clip_value

def limit_scaling_factor(v):
	set_scalar_if_lt(v)
	set_scalar_if_gt(v)

def clip_scalar(inf_norm_q):
	if inf_norm_q>1e4:
		inf_norm_q = 1e4
	elif inf_norm_q< 1e-4:
		inf_norm_q = 1.0
	return inf_norm_q

def mul_diag(sparse_mat, l_vec, r_vec):
	return scipy.sparse.diags(l_vec) * sparse_mat * scipy.sparse.diags(r_vec) 

def scale_prob(qp_problem, iters=0):
	""" Perform Ruiz equilibration on matrix P and A"""
	P_scaled = qp_problem['P']
	A_scaled = qp_problem['A']
	q_scaled = qp_problem['q']
	for i in range(iters):
		D_0 = np.linalg.norm(P_scaled.todense(), ord=np.inf, axis=0)
		D_temp_A = np.linalg.norm(A_scaled.todense(), ord=np.inf, axis=0)
		D_temp = np.maximum(D_0, D_temp_A)
		E_temp = np.linalg.norm(A_scaled.todense(), ord=np.inf, axis=1)

		limit_scaling_factor(D_temp)
		limit_scaling_factor(E_temp)

		D_temp = 1.0/np.sqrt(D_temp)
		E_temp = 1.0/np.sqrt(E_temp)

		P_scaled = mul_diag(P_scaled, D_temp, D_temp)
		A_scaled = mul_diag(A_scaled, E_temp, D_temp)
		q_scaled = D_temp * q_scaled

		D_temp = np.linalg.norm(P_scaled.todense(), ord=np.inf, axis=0)
		c_temp = np.mean(D_temp)

		inf_norm_q = np.linalg.norm(q_scaled, ord=np.inf)
		inf_norm_q = clip_scalar(inf_norm_q)
		
		c_temp = max(c_temp, inf_norm_q)
		c_temp = clip_scalar(c_temp)
		c_temp = 1.0/c_temp
		P_scaled = P_scaled*c_temp

	qp_problem['A'] = A_scaled
	qp_problem['P'] = P_scaled