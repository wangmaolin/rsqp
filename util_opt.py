import numpy as np
import re
import scipy

data_pack_num = 16

def isca_c_to_hbm_pc(isca_c):
	assert isca_c % data_pack_num ==0
	hbm_pc = isca_c//data_pack_num
	return hbm_pc

def nnz_col_split(mat_padded, pack_guide_dict, col_idx_str, isca_c):
	str_pass_for_guide = re.sub(8*'a', 8*'2', col_idx_str)

	str_pass_for_guide = re.sub(4*'b', 4*'4',str_pass_for_guide)
	str_pass_for_guide = re.sub('[ab][ab][ab][ab]', 4*'4',str_pass_for_guide)

	str_pass_for_guide = re.sub(2*'c', 2*'8',str_pass_for_guide)
	str_pass_for_guide = re.sub('[abc][abc]', 2*'8',str_pass_for_guide)

	str_pass_for_guide = re.sub('[abcde]', 'z',str_pass_for_guide)

	""" guide of how to pack each row"""
	pack_guide = [pack_guide_dict.get(x) for x in str_pass_for_guide]

	""" matrix data after packing"""
	cu_dat_pack = []
	""" original nnz col indices in the csr format"""
	origin_col_pack = []

	total_padding = 0
	pack_guide_idx = 0

	for row_idx in range(mat_padded['dim_m']):
		row_start = mat_padded['indptr'][row_idx]
		row_end = mat_padded['indptr'][row_idx+1]
		nnz_indices = mat_padded['indices'][row_start:row_end]
		nnz_data = mat_padded['data'][row_start:row_end]
		nnz_col_cnt = len(nnz_indices)

		if nnz_col_cnt>isca_c:
			pack_guide_idx += np.ceil(nnz_col_cnt/isca_c).astype(np.uint32)
			_, row_padding_number = align_compute_padding(nnz_col_cnt, isca_c)
		else:
			row_pack_guide = pack_guide[pack_guide_idx]
			pack_guide_idx += 1
			assert(row_pack_guide >= nnz_col_cnt)
			row_padding_number = row_pack_guide - nnz_col_cnt

		total_padding += (row_padding_number)
		""" use -1 as the idx of zero padding, the data"""
		bin_width = row_padding_number+nnz_col_cnt
		col_padding = np.ones(bin_width)*-1
		data_padding = np.zeros(bin_width)	

		if nnz_col_cnt <= isca_c//2:
			""" align the col access by mod isca_c for pattern a, b, c"""
			for data_item, col_item in zip(nnz_data,nnz_indices):
				""" check if there is room """
				assert not np.all(col_padding>=0)

				preferred_bin_idx = col_item % bin_width
				search_right=col_padding[preferred_bin_idx:]
				search_left=col_padding[:preferred_bin_idx]
				if np.all(search_right>=0):
					allocate_bin_idx = np.argmax(search_left<0)
				else:
					allocate_bin_idx = np.argmax(search_right<0) + preferred_bin_idx

				data_padding[allocate_bin_idx]=data_item
				col_padding[allocate_bin_idx]=col_item

		else:
			data_padding[:nnz_col_cnt]=nnz_data
			col_padding[:nnz_col_cnt]=nnz_indices

		cu_dat_pack = np.concatenate((cu_dat_pack, data_padding)).astype(np.float32)
		""" pad the -1 to the col idx"""
		origin_col_pack = np.concatenate((origin_col_pack, col_padding)).astype(np.int32)

	assert(len(origin_col_pack)% isca_c == 0)
	# print('original nnz: \t{} \tpaddings: \t{}'.format(len(mat_padded['data']), total_padding))

	return cu_dat_pack, origin_col_pack

def mul_scheduling(col_idx_str, isca_c):
	""" Find aaaaaaaa pattern"""
	str_pass_for_cnt = re.sub(8*'a', '8', col_idx_str)

	""" Find bbbb pattern"""
	str_pass_for_cnt = re.sub(4*'b', '4',str_pass_for_cnt)
	""" Find abbb, abab .. pattern"""
	str_pass_for_cnt = re.sub('[ab][ab][ab][ab]', '4',str_pass_for_cnt)

	""" Find cc pattern"""
	str_pass_for_cnt = re.sub(2*'c', '2',str_pass_for_cnt)
	""" Find ab, ac, bc pattern"""
	str_pass_for_cnt = re.sub('[abc][abc]', '2',str_pass_for_cnt)

	""" Find individual a, b, c, d pattern"""
	str_pass_for_cnt = re.sub('[abcd]', '1',str_pass_for_cnt)

	""" use e to represent a row with more than isca_c """
	str_pass_for_cnt = re.sub('e', '0', str_pass_for_cnt)

	""" number of packings in a row, example nnz isca_c-> 1 packing  """
	cu_cnt_pack =np.array([int(x) for x in str_pass_for_cnt], dtype=np.uint32)

	""" padding cu_cnt to align with HBM PCs"""
	_, cnt_padding_num = align_compute_padding(len(cu_cnt_pack), isca_c)
	cnt_padding = np.zeros(cnt_padding_num,dtype=np.uint32)
	cu_cnt_pack = np.concatenate((cu_cnt_pack, cnt_padding))
	return cu_cnt_pack

def nnz_codebook(isca_c):
	hbm_pc = isca_c_to_hbm_pc(isca_c)

	nnz_char_map={}
	for i in range(isca_c+1):
		if i<= 2*hbm_pc:
			nnz_char_map[i] = 'a'
		elif i<=4*hbm_pc:
			nnz_char_map[i] = 'b'
		elif i<=8*hbm_pc:
			nnz_char_map[i] = 'c'
		else:
			nnz_char_map[i] = 'd'
	return nnz_char_map

def pack_char_to_num(isca_c):
	hbm_pc = isca_c_to_hbm_pc(isca_c)

	pack_guide_dict= {'2': 2*hbm_pc,
                   '4': 4*hbm_pc,
                   '8': 8*hbm_pc,
                   'z': data_pack_num*hbm_pc}
	return pack_guide_dict

def struct_encoding(mat_padded, nnz_char_map, isca_c):
	""" nnz of each row """
	col_idx_list=[]
	for row_idx in range(mat_padded['dim_m']):
		row_start = mat_padded['indptr'][row_idx]
		row_end = mat_padded['indptr'][row_idx+1]
		nnz_col_indices = mat_padded['indices'][row_start:row_end]
		nnz_col_cnt = len(nnz_col_indices)

		"""need to break down rows with more than isca_c non-zeros """
		if nnz_col_cnt >isca_c:
			single_row_repeat_pack=np.ceil(nnz_col_cnt/isca_c).astype(np.uint32)
			""" use e to represent a row with more than isca_c"""
			col_idx_list.append((single_row_repeat_pack-1)*'e'+'d')
		else: 
			cntmap = nnz_char_map.get(nnz_col_cnt)
			assert(cntmap is not None)
			col_idx_list.append(cntmap)
	
	""" nnz2str mapping """
	col_idx_str = ''.join(e for e in col_idx_list)
	return col_idx_str

def skip_condense(vt_bit_map):
	vec_length = vt_bit_map.shape[0]
	vt_loc_map = np.array(range(vec_length), dtype=np.uint32)
	return vt_loc_map

def first_fit_condense(vt_bit_map, isca_c):
	vec_length = vt_bit_map.shape[0]
	""" the location of vector after tetris condensing"""
	vt_loc_map = np.ones(vec_length, dtype=np.uint32)*-1
	condense_bitmap = np.zeros_like(vt_bit_map, dtype=bool)

	for i in range(vec_length):
		bit_map_single= vt_bit_map[i,:]

		bit_map_after_merge = np.logical_and(condense_bitmap, bit_map_single)
		first_fit_loc = np.where(np.any(bit_map_after_merge, axis=1) == False)[0][0]
		vt_loc_map[i]=first_fit_loc
		condense_temp_slice = condense_bitmap[first_fit_loc,:] 
		condense_bitmap[first_fit_loc,:] = np.logical_or(bit_map_single, condense_temp_slice)

	max_loc_after_condense = max(vt_loc_map) + 1

	# tetris_redundency_ratio = max_loc_after_condense * isca_c/vec_length 
	# tetris_density = np.sum(vt_bit_map)/(max_loc_after_condense * isca_c)
	# print('tetris reduendency ratio: \t{:.2f}'.format(tetris_redundency_ratio))
	# print('tetris density ratio: \t\t{:.2f}'.format(tetris_density)) 

	return vt_loc_map

def align_compute_padding(dim_n, isca_c):
	assert isca_c % data_pack_num == 0 and isca_c > 0

	if dim_n%isca_c == 0:
		dim_n_pad_num = 0
	else:
		dim_n_pad_num = isca_c - dim_n%isca_c
	return dim_n + dim_n_pad_num, dim_n_pad_num

def mat_pad(mat_csc, isca_c):
	p_csr = mat_csc.tocsr()
	mat_padded={}

	dim_m = p_csr.shape[0]
	mat_padded['dim_m'], dim_m_pad_num = align_compute_padding(dim_m, isca_c)
	mat_padded['dim_m_pad_num'] = dim_m_pad_num

	mat_padded['data']=p_csr.data
	mat_padded['indices']=p_csr.indices
	indptr_pad_value = p_csr.indptr[-1]
	mat_padded['indptr']= np.pad(p_csr.indptr, 
							  (0, dim_m_pad_num),
							  'constant',
							  constant_values=(0, indptr_pad_value))

	dim_n = p_csr.shape[1]
	mat_padded['dim_n'], dim_n_pad_num = align_compute_padding(dim_n, isca_c)
	mat_padded['dim_n_pad_num'] = dim_n_pad_num
	return mat_padded

def column_align_op(mat_padded, row_idx, isca_c):
	row_start = mat_padded['indptr'][row_idx]
	row_end = mat_padded['indptr'][row_idx+1]
	nnz_col_indices = mat_padded['indices'][row_start:row_end]
	nnz_data = mat_padded['data'][row_start:row_end]
	nnz_col_cnt = len(nnz_col_indices)

	aligned_dict={}
	aligned_dict['data'] = []
	aligned_dict['indices'] = []

	if nnz_col_cnt > isca_c//2:
		""" try to align the non zero data with preferred bank location"""
		indices_align_queue = [ [] for _ in range(isca_c) ]
		data_align_queue = [ [] for _ in range(isca_c) ]
		for idx, (data_item,indice_item) in enumerate(zip(nnz_data, nnz_col_indices)):
			bank_loc = idx % isca_c
			indices_align_queue[bank_loc].append(indice_item)
			data_align_queue[bank_loc].append(data_item)

		align_heights = [ len(q) for q in indices_align_queue]
		sum_heights=np.sum(align_heights)
		sum_padded, nnz_pad_num = align_compute_padding(sum_heights, isca_c)
		average_height = sum_padded//isca_c

		# print(average_height, nnz_pad_num, align_heights)

		redundent_bank_data=[]
		redundent_bank_indices=[]

		""" Phase 1: reduce high banks"""
		for ind_q, data_q in zip(indices_align_queue, data_align_queue):
			for _ in range(len(ind_q)-average_height):
				redundent_bank_indices.append(ind_q.pop(0))
				redundent_bank_data.append(data_q.pop(0))

		wild_card_flags = [True for _ in range(isca_c)]
		wild_card_num = nnz_pad_num

		""" Phase 2: fill low banks"""
		for bank_idx, (ind_q, data_q) in enumerate(zip(indices_align_queue, data_align_queue)):
			if average_height > len(ind_q) and wild_card_num >0 and wild_card_flags[bank_idx]:
				ind_q.append(bank_idx)
				data_q.append(0.0)
				wild_card_flags[bank_idx]=False
				wild_card_num -= 1

			for _ in range(average_height-len(ind_q)):
				ind_q.append(redundent_bank_indices.pop(0))
				data_q.append(redundent_bank_data.pop(0))

		assert len(redundent_bank_data) == 0

		""" Phase 3: export aligned banks"""
		for i in range(average_height):
			aligned_dict['indices'].extend([ q.pop(0) for q in indices_align_queue])
			aligned_dict['data'].extend([ q.pop(0) for q in data_align_queue])

		aligned_dict['flag']=True
	else:
		aligned_dict['indices'].extend(nnz_col_indices.tolist())
		aligned_dict['data'].extend(nnz_data.tolist())
		aligned_dict['flag']=False

	return aligned_dict

def align_columns_with_banks(mat_padded, isca_c):
	""" explore columns shuffling within a row
	for better packing efficiency"""
	mat_aligned = {}
	mat_aligned['dim_m']=mat_padded['dim_m']
	mat_aligned['dim_n']=mat_padded['dim_n']
	mat_aligned['dim_n_pad_num']=mat_padded['dim_n_pad_num']
	mat_aligned['dim_m_pad_num']=mat_padded['dim_m_pad_num']
	aligned_rows_num = 0
	aligned_data = []
	aligned_indices = []
	aligned_indptr = [0]
	for row_idx in range(mat_padded['dim_m']):
		aligned_dict = column_align_op(mat_padded, row_idx, isca_c)

		if aligned_dict['flag']:
			aligned_rows_num += 1

		aligned_data.extend(aligned_dict['data'])	
		aligned_indices.extend(aligned_dict['indices'])
		last_indptr = aligned_indptr[-1]
		aligned_indptr.append(last_indptr+len(aligned_dict['data']))

	mat_aligned['data']=np.array(aligned_data).astype(np.float32)
	mat_aligned['indices']=np.array(aligned_indices).astype(np.int32)
	mat_aligned['indptr']=np.array(aligned_indptr).astype(np.int32)

	# print('Total aligned rows for better packing {}'.format(aligned_rows_num))

	return mat_aligned

def convert_cnt_to_fadd(cu_dict, isca_c):
	assert len(cu_dict['col_idx']) % isca_c == 0
	assert len(cu_dict['cnt']) %isca_c == 0
	col_pack_size = len(cu_dict['col_idx'])//isca_c
	assert col_pack_size <= len(cu_dict['cnt'])

	cu_dict['align_cnt'] = col_pack_size - np.sum(cu_dict['cnt'][:col_pack_size]==0)

	""" fadd_guide array stores # of acc packs in a row """
	fadd_guide = np.zeros_like(cu_dict['cnt']).astype(np.int32)
	fadd_cnt = 0
	partial_cnt = 0

	for i in range(col_pack_size):
		row_cnt = cu_dict['cnt'][i]
		if row_cnt == 0 or row_cnt == 1:
			fadd_cnt += 1
		else:
			fadd_guide[i] = row_cnt << 1

		if row_cnt == 1:
			assert fadd_cnt < ((1<<15)-1)
			if fadd_cnt == 1:
				""" set fadd_flag=0 """
				fadd_guide[i+1-fadd_cnt] = (fadd_cnt<<1) + 0
			else:
				""" set fadd_flag=1 """
				fadd_guide[i+1-fadd_cnt] = (fadd_cnt<<1) + 1
				partial_cnt += 1
			"""clear state to count next row over one pack"""
			fadd_cnt = 0

	""" Add partial cnt info for spmv instruction"""
	cu_dict['partial_cnt'] = partial_cnt

	""" Operate directly on the original col_idx array"""
	col_2d_view = cu_dict['col_idx'].reshape(-1, isca_c)
	for i in range(col_2d_view.shape[0]):
		col_2d_view[i, 0] += fadd_guide[i]<<16

def offset_generation_helper(paat_dict, 
							 P_dict, 
							 A_dict, 
							 At_dict,
							 dict_key,
							 isca_c):
	addr_offset = 0
	paat_dict['offset_P_'+dict_key] = addr_offset

	addr_offset += len(P_dict[dict_key])//isca_c
	paat_dict['offset_A_'+dict_key] = addr_offset

	addr_offset += len(A_dict[dict_key])//isca_c
	paat_dict['offset_At_'+dict_key] = addr_offset

	paat_dict[dict_key]=np.concatenate((P_dict[dict_key],
                                        A_dict[dict_key],
                                        At_dict[dict_key]))

def concat_PAAt_dict(P_dict,
                     A_dict,
                     At_dict,
					 isca_c):
	""" use offset to seperate dict of different matrices"""
	paat_dict = {}
	paat_dict['constraint_vector_length_padded']=A_dict['dim_m']
	paat_dict['constraint_vector_pad_num']=A_dict['dim_m_pad_num']
	paat_dict['solution_vector_length_padded']=A_dict['dim_n']
	paat_dict['solution_vector_pad_num']=A_dict['dim_n_pad_num']

	""" mat_dat and col_idx can just do plain concat,
	these 2 also shared the same offset & length"""
	offset_generation_helper(paat_dict, P_dict, A_dict, At_dict, 'mat_dat', isca_c)

	""" convert col_idx to fadd + cnt array in place"""
	convert_cnt_to_fadd(P_dict, isca_c)
	convert_cnt_to_fadd(A_dict, isca_c)
	convert_cnt_to_fadd(At_dict, isca_c)

	paat_dict['partial_cnt_P']= P_dict['partial_cnt']
	paat_dict['partial_cnt_A']= A_dict['partial_cnt']
	paat_dict['partial_cnt_At']= At_dict['partial_cnt']

	paat_dict['align_cnt_P']= P_dict['align_cnt']
	paat_dict['align_cnt_A']= A_dict['align_cnt']
	paat_dict['align_cnt_At']= At_dict['align_cnt']

	paat_dict['col_pack_size_P'] = len(P_dict['col_idx'])//isca_c
	paat_dict['col_pack_size_A'] = len(A_dict['col_idx'])//isca_c
	paat_dict['col_pack_size_At'] = len(At_dict['col_idx'])//isca_c

	offset_generation_helper(paat_dict, P_dict, A_dict, At_dict, 'col_idx', isca_c)
	""" concat location map"""
	offset_generation_helper(paat_dict, P_dict, A_dict, At_dict, 'location_map', isca_c)

	""" tetris height for different matrices"""
	paat_dict['tetris_height_P']= max(P_dict['location_map'])+1
	paat_dict['tetris_height_A']= max(A_dict['location_map'])+1
	paat_dict['tetris_height_At']= max(At_dict['location_map'])+1

	""" concat duplicate map"""
	offset_generation_helper(paat_dict, P_dict, A_dict, At_dict, 'duplicate_map', isca_c)

	return paat_dict

def compute_reduced_kkt_mat(qp_problem, sigma, rho):
	reduced_kkt = qp_problem['P'] +\
		sigma * scipy.sparse.identity(qp_problem['P'].shape[0]) +\
		rho * qp_problem['A'].transpose() * qp_problem['A']

	# print('P, A, At nnz: \t{} \tKKT nnz: \t{}'.format(len(qp_problem['P'].data) + 2*len(qp_problem['A'].data), len(reduced_kkt.data)))

	return reduced_kkt

def compute_precond_diag(qp_problem, sigma, rho):
	""" compute the 2 parts for preconditioner computation"""
	solution_vector_length_padded = qp_problem['P'].shape[0]

	precond_diag={}
	precond_diag['part1'] = np.zeros(solution_vector_length_padded)
	precond_diag['part2'] = np.zeros(solution_vector_length_padded)

	PsI = qp_problem['P'] +\
		sigma * scipy.sparse.identity(qp_problem['P'].shape[0])
	AtA = qp_problem['A'].transpose() * qp_problem['A']

	for i in range(solution_vector_length_padded):
		assert (PsI[i,i] + rho*AtA[i,i]) !=0.0
		precond_diag['part1'][i]=PsI[i,i]
		precond_diag['part2'][i]=AtA[i,i]

	return precond_diag
