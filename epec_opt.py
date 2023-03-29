import numpy as np

from util_opt import skip_condense
from util_opt import first_fit_condense

from rsqp_util import struct_encoding
from rsqp_util import ceil_struct
from rsqp_util import mul_scheduling
from rsqp_util import nnz_col_split
from rsqp_util import char2nnz
from rsqp_util import nnz2char

import re
import numpy as np

def reduce_ep_ec(mat_padded, isca_c, arch_code):
	""" turn the nnz in each row to a string
	power of 2 map to a,b,c, ...,d and mapping is fixed regardless of isca_c
	"""
	col_idx_str = struct_encoding(mat_padded, isca_c)
	col_idx_str = ceil_struct(col_idx_str, arch_code)

	""" Next find arch code based on col_idx_str, 
	now just use given arch-code"""

	""" scheduling the nnz to multipliers, used in cvt_cnt2fadd"""
	cu_cnt_pack = mul_scheduling(col_idx_str, isca_c, arch_code)

	""" split input to different multipliers """	
	cu_dat_pack, origin_col_pack = nnz_col_split(mat_padded, 
											  	col_idx_str, 
											  	isca_c,
											  	arch_code)

	""" Split vector into bins for parallel access within one clock cycle"""
	csr_col_access = origin_col_pack.reshape(-1, isca_c)

	""" build the bitmap first, then do the tetris move to condense it"""
	""" the bit map tells which bin each vector needs to be in"""
	vt_bit_map = np.zeros((mat_padded['dim_n'], isca_c), dtype=bool)

	for pack_idx in range(csr_col_access.shape[0]):
		vec_access = csr_col_access[pack_idx, :] 
		for bin_idx in range(isca_c): 
			v_idx = vec_access[bin_idx] 
			if v_idx >=0:
				vt_bit_map[v_idx][bin_idx]=1

	"""make sure tetris can provide isca_c items of the vector 
	when the vector are accessed sequentially"""
	for i in range(mat_padded['dim_n']):
		bin_idx_when_sequence_access = i % isca_c
		vt_bit_map[i,bin_idx_when_sequence_access] = 1

	""" tetris condensing"""
	bit_map_density = np.sum(vt_bit_map)/vt_bit_map.size
	if bit_map_density > 0.5:
		condense_info_str = "Give Up Compression"
		vt_loc_map = skip_condense(vt_bit_map)
	else:
		condense_info_str = "First Fit Compression"
		vt_loc_map = first_fit_condense(vt_bit_map, isca_c)
	# print(condense_info_str + "\t| bit map density: **** {:.2f} ****".format(bit_map_density))

	""" translate original spmv vector access to tetris structure access"""
	tetris_addr_translation= np.zeros_like(csr_col_access).astype(np.uint32)
	for i in range(tetris_addr_translation.shape[0]):
		for j in range(isca_c):
			original_vec_addr = csr_col_access[i, j]
			if original_vec_addr < 0:
				tetris_addr_translation[i, j] = 0	
			else:
				tetris_addr_translation[i, j] = vt_loc_map[original_vec_addr]	
	
	"""Step 2. update bit maps, requires another data structure
	"""
	tetris_height = max(vt_loc_map)+1
	duplicate_guide = np.zeros((tetris_height, isca_c)).astype(np.uint32)
	for i in range(mat_padded['dim_n']):
		loc = vt_loc_map[i]
		bit_map_to_duplicate = vt_bit_map[i]
		for j in range(isca_c):
			if bit_map_to_duplicate[j]==1:
				duplicate_guide[loc][j] = i%isca_c

	""" cnt, dat, and col pack are results passed back to HW"""
	return {'cnt':cu_cnt_pack.flatten(),  
         	'mat_dat': cu_dat_pack,
         	'location_map': vt_loc_map.flatten(),
		 	'duplicate_map':duplicate_guide.flatten(),
		 	'col_idx': tetris_addr_translation.flatten(),
			'dim_m': mat_padded['dim_m'],
			'dim_n': mat_padded['dim_n'],
			'dim_n_pad_num': mat_padded['dim_n_pad_num'],
			'dim_m_pad_num': mat_padded['dim_m_pad_num']
	}

def calc_reduction_ratio(col_idx_str, isca_c, arch_code):
	place_holder = '$'
	str_pass_for_guide = ceil_struct(col_idx_str, arch_code)
	for item in range(len(arch_code)):
		char_to_find = arch_code[item]

		cnt_using_all_pc = isca_c//char2nnz(char_to_find, isca_c)

		str_pass_for_guide = re.sub(cnt_using_all_pc*char_to_find, 
							place_holder, 
							str_pass_for_guide)

		re_to_find = cnt_using_all_pc*('['+arch_code[:item+1]+']')

		str_pass_for_guide = re.sub(re_to_find, 
							place_holder, 
							str_pass_for_guide)

	return (len(col_idx_str)-len(str_pass_for_guide))/len(col_idx_str), len(str_pass_for_guide)

def take_second_from_tuple(elem):
	""" help function for ranking list"""
	return elem[1]

def suggest_arch_code(col_idx_str, isca_c):
	""" suggest arch code based on sparisty pattern,
		rank mode by the length of total reduction """

	full_arch_code ='abcdefgh'
	end_arch_code = nnz2char(isca_c)
	end_pos = full_arch_code.find(end_arch_code)
	arch_code_candidate= full_arch_code[:end_pos+1]
	place_holder = '$'
	ratio_rank = []

	for item in range(len(arch_code_candidate)-1):
		char_to_find = arch_code_candidate[item]

		cnt_using_all_pc = isca_c//char2nnz(char_to_find, isca_c)

		str_pass_for_guide = re.sub(cnt_using_all_pc*char_to_find, 
							place_holder, 
							col_idx_str)

		re_to_find = cnt_using_all_pc*('['+arch_code_candidate[:item+1]+']')

		str_pass_for_guide = re.sub(re_to_find, 
							place_holder, 
							str_pass_for_guide)

		reduction_ratio = (len(col_idx_str)-len(str_pass_for_guide))/len(col_idx_str)

		ratio_rank.append((char_to_find, reduction_ratio))

	""" rank the reduction ratio"""
	ratio_rank.sort(reverse=True, key=take_second_from_tuple)
	arch_suggestion = arch_code_candidate[-1]
	suggest_list = []
	for item in ratio_rank:
		arch_suggestion += item[0]
		suggest_list.append(''.join(sorted(arch_suggestion)))

	return suggest_list
