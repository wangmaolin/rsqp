import numpy as np
from scipy import sparse
from util_opt import data_pack_num

class InstructionSet:
    def __init__(self,
                 max_data_size,
                 hbm_pc):
        self.cu_inst_num = 0

        self.hbm_pc = hbm_pc
        self.isca_c = hbm_pc * data_pack_num
        self.inst_halt = 0
        self.inst_branch = 1
        self.inst_complete_tetris = 2
        self.inst_spmv = 3
        self.inst_axpby= 4
        self.inst_load_tetris = 5
        self.inst_dot = 6
        self.inst_scalar_op = 7
        self.inst_norm_inf = 8

        self.uint32_pack_num = data_pack_num
        self.indice_mem_words = 0
        self.uint32_data_region = np.zeros(max_data_size, dtype=np.uint32)
        self.uint32_data_pointer = 0
        self.stage_size_recorder = []

        self.col_mem_words = 0
        self.col_region = []
        for i in range(self.hbm_pc):
            self.col_region.append(np.zeros(max_data_size, dtype=np.uint32))
        self.col_region_pointer = 0

        self.nnz_mem_words = 0
        self.nnz_data_region = []
        for i in range(self.hbm_pc):
            self.nnz_data_region.append(np.zeros(max_data_size, dtype=np.float32))
        self.nnz_data_region_pointer = 0

        self.lhs_mem_words = 0
        self.lhs_data_region = []
        for _ in range(self.hbm_pc):
            self.lhs_data_region.append(np.zeros(max_data_size, dtype=np.float32))
        self.lhs_data_region_pointer = 0

        self.rhs_mem_words = 0
        self.rhs_data_region = []
        for _ in range(self.hbm_pc):
            self.rhs_data_region.append(np.zeros(max_data_size, dtype=np.float32))
        self.rhs_data_region_pointer = 0

        self.mem_ground_truth_loc = 0
        self.mem_verify_loc = 0
        self.verify_vec_pack_len = 0

        self.program_info = np.zeros((4, 4), dtype=np.uint32)
        self.info_ptr = 0

        self.sol_vec_pack_len = -1
        self.con_vec_pack_len = -1
        self.unified_vec_pack_len = -1

        """
        program info are 128 bits aligned
            - [0]: file header magic number: 2135247942
            - [1]: inst_rom_pack_size
            - [2]:
            - [3]:

            - [4]:
            - [5]: stage_5_pack_size or indice mem pack words
            - [6]:
            - [7]: mem data words

            - [8]: mem ground truth loc
            - [9]: mem verify loc
            - [10]: verify vec pack len
            - [11]: mem sol words

            - [12]: HBM PC NUM
            - [13]: col_mem_words
            - [14]: nnz_mem_words
            - [15]: lhs_mem_words/rhs_mem_words assert equal
        """

    def codegen(self, file_name):
        """ make sure the added mem indice contents are aligned """
        assert self.cu_inst_num % 4 == 0

        for item in self.stage_size_recorder:
            assert(item % 16 == 0)

        inst_rom_pack_size = self.stage_size_recorder[0]//data_pack_num

        """ set program info """
        self.add_info([2135247942,
                       inst_rom_pack_size,
                       0,
                       0])

        self.add_info([0,
                       inst_rom_pack_size+2,
                       0,
                       0])

        self.add_info([self.mem_ground_truth_loc,
                       self.mem_verify_loc,
                       self.verify_vec_pack_len,
                       0])

        self.add_info([self.hbm_pc,
                       self.col_mem_words,
                       self.nnz_mem_words,
                       self.lhs_mem_words])

        with open(file_name, "wb") as f:
            """ Write program meta information"""
            self.program_info.tofile(f)
            """ Write to mem_indice """
            self.uint32_data_region[0:self.uint32_data_pointer].tofile(f)
            """ Write to mem_col"""
            for i in range(self.hbm_pc):
                self.col_region[i][0:self.col_region_pointer].tofile(f)
            """ Write to mem_nnz"""
            for i in range(self.hbm_pc):
                self.nnz_data_region[i][0:self.nnz_data_region_pointer].tofile(f)

            """ Write to mem_lhs"""
            for i in range(self.hbm_pc):
                self.lhs_data_region[i][0:self.lhs_data_region_pointer].tofile(f)
            """ Write to mem_rhs"""
            for i in range(self.hbm_pc):
                self.rhs_data_region[i][0:self.rhs_data_region_pointer].tofile(f)

    def add_axpby_inst(self,
                       alpha_addr,
                       beta_addr,
                       src_tetris_addr,
                       src_vf_addr,
                       dst_sel_tetris,
                       dst_addr,
                       pack_size,
                       op_type = 0,
                       program_end=False):
        """ generate the instruction for ax plus by instruction """
        op0 = self.inst_axpby
        op1 = alpha_addr +\
            (beta_addr<<6) +\
            (src_tetris_addr<<12) +\
            (src_vf_addr<<18) +\
            (dst_addr<<24) +\
            (dst_sel_tetris<<30)

        op2 = op_type
        op3 = pack_size
        self.add_inst([op0, op1, op2, op3], program_end)

        # """ To be in emit_ir and pass to link_binary.py """
        # return [op0, op1, op2, op3]

    def add_complete_tetris_inst(self,
                             tetris_height,
                             duplicate_map_offset,
                             program_end=False):
        op0=self.inst_complete_tetris
        op1=tetris_height
        op2=duplicate_map_offset
        self.add_inst([op0, op1, op2, 0], program_end)

    def add_load_tetris_inst(self,
                       src_addr,
                       src_sel_lhs,
                       location_map_offset,
                       pack_size,
                       program_end=False):
        """ generate the instruction for ax plus by instruction """
        op0 = self.inst_load_tetris
        op1 = src_addr+\
            (src_sel_lhs<<6)
        op2 = location_map_offset
        op3 = pack_size
        self.add_inst([op0, op1, op2, op3], program_end)

    def add_norm_inf_inst(self,
                       src_addr,
                       dst_addr,
                       src_sel_lhs,
                       pack_size,
                       program_end=False):
        """ generate the instruction for ax plus by instruction """
        op0 = self.inst_norm_inf
        op1 = src_addr+\
            (dst_addr<<6)+\
            (src_sel_lhs<<12)
        op2 = pack_size
        self.add_inst([op0, op1, op2, 0], program_end)

    def compute_vecbuf_base_loc(self,
                                vecbuf_addr):

        return vecbuf_addr * self.unified_vec_pack_len

    def add_dot_inst(self,
                     sel_norm,
                     src_tetris_addr,
                     src_vf_addr,
                     dst_reg,
                     pack_size,
                     program_end=False):
        """ generate the instruction for dot instruction """
        op0 = self.inst_dot
        op1 = src_tetris_addr +\
            (src_vf_addr<<6) +\
            (dst_reg<<12)+\
            (sel_norm<<18)
        op2 = pack_size
        self.add_inst([op0, op1, op2, 0], program_end)

    def add_scalar_op_inst(self,
                           src_0_reg,
                           src_1_reg,
                           scalar_op,
                           dst_reg,
                           imme_flag=0,
                           program_end=False):
        """ generate the instruction for dot instruction """
        op0 = self.inst_scalar_op
        op1 = src_0_reg+\
            (src_1_reg<<6) +\
            (dst_reg<<12) +\
            (scalar_op<<18) +\
            (imme_flag<<22)
        self.add_inst([op0, op1, 0, 0], program_end)

    def add_branch_inst(self,
                        src_0_reg,
                        src_1_reg,
                        jump_address,
                        imme_flag=0,
                        imme_number=0,
                        program_end=False):
        """ generate the instruction for dot instruction """
        op0 = self.inst_branch
        op1 = src_0_reg+\
            (src_1_reg<<6)+\
            (imme_flag<<12)
        self.add_inst([op0, op1, jump_address, imme_number], program_end)

    def add_spmv_inst(self,
                      mat_id,
                      dst_addr,
                      dst_sel_lhs,
                      program_end=False):
        op0 = self.inst_spmv
        op1 = mat_id
        op2 = dst_addr+\
            (dst_sel_lhs <<6)
        self.add_inst([op0, op1, op2, 0], program_end)

    def add_inst(self, op_list, program_end = False):
        self.add_vector_uint32(op_list, program_end)
        self.cu_inst_num +=1

    def add_vector_uint32(self, vec, record_stage_size = False):
        size = len(vec)
        assert size % 4 ==0
        self.uint32_data_region[self.uint32_data_pointer: self.uint32_data_pointer+ size]=vec
        self.uint32_data_pointer += size
        self.indice_mem_words += size
        if record_stage_size:
            self.stage_size_recorder.append(self.uint32_data_pointer)

    def add_external_inst(self, vec):
        """ add the binary compiled by external compiler """
        size = len(vec)
        assert size % 4 ==0
        self.uint32_data_region[self.uint32_data_pointer: self.uint32_data_pointer+ size]=vec
        self.uint32_data_pointer += size
        self.indice_mem_words += size
        self.cu_inst_num += size // 4

    def add_info(self,  info_list):
        for idx, info in enumerate(info_list):
            self.program_info[self.info_ptr][idx]=info
        self.info_ptr += 1

    def set_verify_info(self, mem_ground_truth_loc, mem_verify_loc, verify_vec_pack_len):
        self.mem_ground_truth_loc = mem_ground_truth_loc
        self.mem_verify_loc = mem_verify_loc
        self.verify_vec_pack_len = verify_vec_pack_len

    def add_vector_col_mem(self, vec):
        """ split the col index into different HBM PCs"""
        size = len(vec)
        assert size % self.isca_c == 0
        vec_stride = vec.reshape(-1, data_pack_num)
        size_each_pc = size//self.hbm_pc
        region_start = self.col_region_pointer
        self.col_region_pointer += size_each_pc
        region_end = self.col_region_pointer
        for i in range(self.hbm_pc):
            self.col_region[i][region_start:region_end] = np.concatenate(vec_stride[i::self.hbm_pc, :])

        self.col_mem_words += size_each_pc

    def add_vector_nnz_mem(self, vec):
        """ split the matrix nnz into different HBM PCs"""
        size = len(vec)
        assert size % self.isca_c == 0
        vec_stride = vec.reshape(-1, data_pack_num)
        size_each_pc = size//self.hbm_pc
        region_start = self.nnz_data_region_pointer
        self.nnz_data_region_pointer += size_each_pc
        region_end = self.nnz_data_region_pointer
        for i in range(self.hbm_pc):
            self.nnz_data_region[i][region_start:region_end] = np.concatenate(vec_stride[i::self.hbm_pc, :])

        self.nnz_mem_words += size_each_pc

    def add_vector_lhs_mem(self, vec):
        """ split the lhs vector into different HBM PCs"""
        vec=self.unified_vector_container(vec)

        size = len(vec)
        assert size % self.isca_c == 0
        vec_stride = vec.reshape(-1, data_pack_num)
        size_each_pc = size//self.hbm_pc
        region_start = self.lhs_data_region_pointer
        self.lhs_data_region_pointer += size_each_pc
        region_end = self.lhs_data_region_pointer
        for i in range(self.hbm_pc):
            self.lhs_data_region[i][region_start:region_end] = np.concatenate(vec_stride[i::self.hbm_pc, :])

        self.lhs_mem_words += size_each_pc

    def add_vector_rhs_mem(self, vec):
        """ split the rhs vector into different HBM PCs"""
        vec=self.unified_vector_container(vec)

        size = len(vec)
        assert size % self.isca_c == 0
        vec_stride = vec.reshape(-1, data_pack_num)
        size_each_pc = size//self.hbm_pc
        region_start = self.rhs_data_region_pointer
        self.rhs_data_region_pointer += size_each_pc
        region_end = self.rhs_data_region_pointer
        for i in range(self.hbm_pc):
            self.rhs_data_region[i][region_start:region_end] = np.concatenate(vec_stride[i::self.hbm_pc, :])

        self.rhs_mem_words += size_each_pc

    def unified_vector_container(self, vec):
        container_size = self.unified_vec_pack_len * self.isca_c
        assert container_size > 0 and container_size >= len(vec)
        vector_container = np.zeros(container_size)
        vector_container[:len(vec)]=vec
        return vector_container
