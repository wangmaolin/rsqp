import numpy as np
import re
from util_opt import data_pack_num
from util_opt import align_compute_padding

def nnz2char(nnz_col_cnt):
	""" 1->a, 2->b, 4->c, ...,"""
	if nnz_col_cnt == 0:
		char_map = 'a'
	else:
		alphabet_order = np.ceil(np.log2(nnz_col_cnt)).astype(int)
		char_map=chr(97+alphabet_order)

	return char_map

def char2nnz(char_map, isca_c):
	""" a->1, b->2, c->4, ..."""
	assert char_map in {'a', 'b', 'c','d','e','f','g','h','$'}
	if char_map == '$':
		nnz_col_cnt = isca_c
	else:
		nnz_col_cnt = 2**(ord(char_map)-ord('a'))
	return nnz_col_cnt

def nnz_col_split(mat_padded, col_idx_str, isca_c, arch_code):
	str_pass_for_guide = col_idx_str
	for item in range(len(arch_code)):
		char_to_find = arch_code[item]
		char_num = char2nnz(char_to_find, isca_c)
		log_num = int(np.log2(char_num))

		cnt_using_all_pc = isca_c//char2nnz(char_to_find, isca_c)

		str_pass_for_guide= re.sub(cnt_using_all_pc*char_to_find, 
                            cnt_using_all_pc*str(log_num), 
							str_pass_for_guide)

		re_to_find = cnt_using_all_pc*('['+arch_code[:item+1]+']')

		str_pass_for_guide = re.sub(re_to_find, 
                            cnt_using_all_pc*str(log_num), 
							str_pass_for_guide)

	str_pass_for_guide = re.sub('\$', 
                            str(int(np.log2(isca_c))), 
							str_pass_for_guide)

	"""
	str_pass_for_guide = re.sub(8*'a', 8*'2', col_idx_str)

	str_pass_for_guide = re.sub(4*'b', 4*'4',str_pass_for_guide)
	str_pass_for_guide = re.sub('[ab][ab][ab][ab]', 4*'4',str_pass_for_guide)

	str_pass_for_guide = re.sub(2*'c', 2*'8',str_pass_for_guide)
	str_pass_for_guide = re.sub('[abc][abc]', 2*'8',str_pass_for_guide)

	# str_pass_for_guide = re.sub('[abcde]', 'z',str_pass_for_guide)
	str_pass_for_guide = re.sub('[abcd$]', 'z',str_pass_for_guide)
	"""

	""" guide of how to pack each row"""
	pack_guide = [2**int(x) for x in str_pass_for_guide]

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

def valid_arch_code(arch_code, isca_c):
	""" check arch_code valid"""
	assert isca_c == char2nnz(arch_code[-1], isca_c)
	arch_code_ord_list = [ ord(m) for m in arch_code ]
	assert np.all(np.diff(arch_code_ord_list))

def ceil_struct(col_idx_str, arch_code):
	""" upscale chars not in the arch_code dict"""
	ceil_string = col_idx_str
	start_char='a'
	for item in range(len(arch_code)):
		char_to_ceil = arch_code[item]

		for i in range(ord(start_char),ord(char_to_ceil)):
			""" replace small char to the nearby char in arch_code"""
			# print("{} -> {}".format(chr(i), char_to_ceil))
			ceil_string = re.sub(chr(i), char_to_ceil, ceil_string)

		start_char=chr(ord(char_to_ceil)+1)

	return ceil_string

def cnt_char_to_num(cnt_char):
	if cnt_char == '$':
		return 0
	else:
		return 2**int(cnt_char)

def mul_scheduling(col_idx_str, isca_c, arch_code):
	""" use power of 2 cnt, 8 is 2^3 and so on"""
	str_pass_for_cnt = col_idx_str
	for item in range(len(arch_code)):
		char_to_find = arch_code[item]
		cnt_using_all_pc = isca_c//char2nnz(char_to_find, isca_c)
		log_cnt = int(np.log2(cnt_using_all_pc))
		# print(cnt_using_all_pc*char_to_find, cnt_using_all_pc, str(log_cnt), cnt_char_to_num(str(log_cnt)))
		str_pass_for_cnt = re.sub(cnt_using_all_pc*char_to_find, 
                            str(log_cnt), 
							str_pass_for_cnt)
		re_to_find = cnt_using_all_pc*('['+arch_code[:item+1]+']')
		# print(re_to_find)
		str_pass_for_cnt = re.sub(re_to_find, 
                            str(log_cnt), 
							str_pass_for_cnt)

	""" number of packings in a row, example nnz isca_c-> 1 packing  """
	cu_cnt_pack =np.array([cnt_char_to_num(x) for x in str_pass_for_cnt], dtype=np.uint32)

	""" padding cu_cnt to align with HBM PCs"""
	_, cnt_padding_num = align_compute_padding(len(cu_cnt_pack), isca_c)
	cnt_padding = np.zeros(cnt_padding_num,dtype=np.uint32)
	cu_cnt_pack = np.concatenate((cu_cnt_pack, cnt_padding))

	return cu_cnt_pack

def struct_encoding(mat_padded, isca_c):
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
			# col_idx_list.append((single_row_repeat_pack-1)*'e'+'d')
			full_pad_char = nnz2char(isca_c)
			col_idx_list.append((single_row_repeat_pack-1)*'$'+full_pad_char)
		else: 
			# cntmap = nnz_char_map.get(nnz_col_cnt)
			cntmap = nnz2char(nnz_col_cnt)
			assert(cntmap is not None)
			col_idx_list.append(cntmap)
	
	""" nnz2str mapping """
	col_idx_str = ''.join(e for e in col_idx_list)
	return col_idx_str