import numpy as np
from util_opt import data_pack_num
from rsqp_util import char2nnz

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--arch-code',type=str, required=True)
parser.add_argument('--hbm-pc', type=int, required=True)
parser.add_argument('--max-height', type=int, required=True)
parser.add_argument('--max-vec-len', type=int, required=True)
parser.add_argument('--output-dir',type=str,  required=True)

parser.add_argument('--verify', type=int, default=0)
parser.add_argument('--select-bram', type=int, default=0)
parser.add_argument('--fadd-ii', type=int, default=8)

def hw_gen(arch_code, 
           hbm_pc,
           tetris_height,
           uni_vec_pack_len,
           fadd_ii,
           ram_type,
           csim_verify,
           output_dir):
    """ The meta program that generates the hardware """
    assert hbm_pc in {1, 2, 4, 8}
    src_root= output_dir + '/src/'

    snippet_file = src_root + 'arch_cfg.h'
    with open(snippet_file, "w") as text_file:
        if csim_verify:
            text_file.write("#define REGISTER_DEBUG\n")
            # text_file.write("#define SW_EMU_DEBUG\n")
        assert fadd_ii in {4, 8, 16}
        text_file.write("#define FADD_SPLIT {}\n".format(fadd_ii))
        text_file.write("#define FADD_LB {}\n".format(int(np.log2(fadd_ii))))
        text_file.write("#define HBM_PC {}\n".format(hbm_pc))
        text_file.write("#define TETRIS_HEIGHT {}\n".format(tetris_height))
        text_file.write("#define UNI_VEC_PACK_SIZE {}\n".format(uni_vec_pack_len))

    snippet_file = src_root +  'top_interface.h'
    with open(snippet_file, "w") as text_file:
        for i in range(hbm_pc):
            text_file.write("data_pack_t *mem_nnz_{},\n".format(i))

        for i in range(hbm_pc):
            text_file.write("indice_pack_t *mem_col_{},\n".format(i))

        for i in range(hbm_pc):
            text_file.write("data_pack_t *mem_lhs_{},\n".format(i))
            text_file.write("data_pack_t *mem_rhs_{},\n".format(i))

    snippet_file = src_root +  'mem_lr_interface.h'
    with open(snippet_file, "w") as text_file:
        for i in range(hbm_pc):
            text_file.write("data_pack_t *mem_lhs_{},\n".format(i))
            text_file.write("data_pack_t *mem_rhs_{},\n".format(i))

    snippet_file = src_root + 'mem_lr_func_call.h'
    with open(snippet_file, "w") as text_file:
        for i in range(hbm_pc):
            text_file.write("mem_lhs_{0},\n".format(i))
            text_file.write("mem_rhs_{0},\n".format(i))

    snippet_file = src_root + 'gmem_bundle.h'
    with open(snippet_file, "w") as text_file:
        for i in range(hbm_pc):
            text_file.write("#pragma HLS INTERFACE m_axi port = mem_nnz_{0} offset = slave bundle = gmem{1} depth = 200000\n".format(i, i+1))
            text_file.write("#pragma HLS INTERFACE m_axi port = mem_lhs_{0} offset = slave bundle = gmem{1} depth = 200000\n".format(i, i+1))

        for i in range(hbm_pc):
            text_file.write("#pragma HLS INTERFACE m_axi port = mem_col_{0} offset = slave bundle = gmem{1} depth = 200000\n".format(i, hbm_pc + i + 1))
            text_file.write("#pragma HLS INTERFACE m_axi port = mem_rhs_{0} offset = slave bundle = gmem{1} depth = 200000\n".format(i, hbm_pc + i + 1))

    snippet_file = src_root + 'top_func_call.h'
    with open(snippet_file, "w") as text_file:
        for i in range(hbm_pc):
            text_file.write("mem_nnz_{0},\n".format(i))

        for i in range(hbm_pc):
            text_file.write("mem_col_{0},\n".format(i))

        for i in range(hbm_pc):
            text_file.write("mem_lhs_{0},\n".format(i))
            text_file.write("mem_rhs_{0},\n".format(i))

    snippet_file = src_root + 'tb_array_create.h'
    with open(snippet_file, "w") as text_file:
        for i in range(hbm_pc):
            text_file.write("data_pack_t* mem_nnz_{} = new data_pack_t[DATA_MEM_DEPTH];\n".format(i))
            text_file.write("indice_pack_t* mem_col_{} = new indice_pack_t[INDICE_MEM_DEPTH];\n".format(i))
            text_file.write("data_pack_t* mem_lhs_{} = new data_pack_t[VB_MEM_DEPTH];\n".format(i))
            text_file.write("data_pack_t* mem_rhs_{} = new data_pack_t[VB_MEM_DEPTH];\n".format(i))

    snippet_file = src_root + 'tb_nnz_value_copy.h'
    with open(snippet_file, "w") as text_file:
        for i in range(hbm_pc):
            text_file.write("inst_file_stream.read(reinterpret_cast<char *>(mem_nnz_{}), nnz_mem_bytes);\n".format(i))

    snippet_file = src_root + 'tb_col_value_copy.h'
    with open(snippet_file, "w") as text_file:
        for i in range(hbm_pc):
            text_file.write("inst_file_stream.read(reinterpret_cast<char *>(mem_col_{}), col_mem_bytes);\n".format(i))

    snippet_file = src_root + 'tb_lr_value_init.h'
    with open(snippet_file, "w") as text_file:
        for i in range(hbm_pc):
            text_file.write("inst_file_stream.read(reinterpret_cast<char *>(mem_lhs_{}), lr_mem_bytes);\n".format(i))
            # text_file.write("memset(mem_lhs_{}, 0, sizeof(data_pack_t));\n".format(i))

        for i in range(hbm_pc):
            text_file.write("inst_file_stream.read(reinterpret_cast<char *>(mem_rhs_{}), lr_mem_bytes);\n".format(i))
            # text_file.write("memset(mem_rhs_{}, 0, sizeof(data_pack_t));\n".format(i))

    snippet_file = src_root + 'read_hbm.h'
    with open(snippet_file, "w") as text_file:
        for i in range(hbm_pc):
            text_file.write("data_pack_t temp_nnz_{0} = mem_nnz_{0}[mat_addr + loc];\n".format(i))
            text_file.write("indice_pack_t temp_col_{0} = mem_col_{0}[col_addr + loc];\n".format(i))

    snippet_file = src_root + 'unpack_hbm.h'
    with open(snippet_file, "w") as text_file:
        for i in range(hbm_pc):
            text_file.write("tmpIn1[{0}*DATA_PACK_NUM+j]=temp_nnz_{0}.data[j];\n".format(i))

        for i in range(hbm_pc):
            text_file.write("tmpCol[{0}*DATA_PACK_NUM+j]=temp_col_{0}.data[j];\n".format(i))

    snippet_file = src_root + 'lr_depend_disable.h'
    with open(snippet_file, "w") as text_file:
        for i in range(hbm_pc):
            text_file.write("#pragma HLS DEPENDENCE variable=mem_lhs_{0} type=INTER direction=RAW dependent=FALSE\n".format(i))
            text_file.write("#pragma HLS DEPENDENCE variable=mem_rhs_{0} type=INTER direction=RAW dependent=FALSE\n".format(i))


    snippet_file = src_root + 'read_lr_hbm.h'
    with open(snippet_file, "w") as text_file:
        for i in range(hbm_pc):
            text_file.write("data_pack_t temp_lhs_{0} = mem_lhs_{0}[left_read_base_loc + loc];\n".format(i))
            text_file.write("data_pack_t temp_rhs_{0} = mem_rhs_{0}[right_read_base_loc + loc];\n".format(i))

    snippet_file = src_root + 'dot_read_lr_hbm.h'
    with open(snippet_file, "w") as text_file:
        for i in range(hbm_pc):
            text_file.write("data_pack_t temp_lhs_{0} = mem_lhs_{0}[left_read_base_loc + loc+i];\n".format(i))
            text_file.write("data_pack_t temp_rhs_{0} = mem_rhs_{0}[right_read_base_loc + loc+i];\n".format(i))


    snippet_file = src_root + 'declare_wb_pack.h'
    with open(snippet_file, "w") as text_file:
        for i in range(hbm_pc):
            text_file.write("data_pack_t temp_result_{0};\n".format(i))

    snippet_file = src_root + 'unpack_lr_hbm.h'
    with open(snippet_file, "w") as text_file:
        for i in range(hbm_pc):
            text_file.write("x[{0}*DATA_PACK_NUM+j]=temp_lhs_{0}.data[j];\n".format(i))
            text_file.write("y[{0}*DATA_PACK_NUM+j]=temp_rhs_{0}.data[j];\n".format(i))

    snippet_file = src_root + 'unpack_temp_result_hbm.h'
    with open(snippet_file, "w") as text_file:
        for i in range(hbm_pc):
            text_file.write("dst[{0}*DATA_PACK_NUM+j]=temp_result_{0}.data[j];\n".format(i))

    snippet_file = src_root + 'pack_lr_hbm.h'
    with open(snippet_file, "w") as text_file:
        for i in range(hbm_pc):
            text_file.write("temp_result_{0}.data[j]=dst[{0}*DATA_PACK_NUM+j];\n".format(i))

    snippet_file = src_root + 'write_lhs_hbm.h'
    with open(snippet_file, "w") as text_file:
        for i in range(hbm_pc):
            text_file.write("mem_lhs_{0}[write_base_loc + loc]=temp_result_{0};\n".format(i))

    snippet_file = src_root + 'write_rhs_hbm.h'
    with open(snippet_file, "w") as text_file:
        for i in range(hbm_pc):
            text_file.write("mem_rhs_{0}[write_base_loc + loc]=temp_result_{0};\n".format(i))

    snippet_file = src_root + 'read_lhs_hbm.h'
    with open(snippet_file, "w") as text_file:
        for i in range(hbm_pc):
            text_file.write("temp_result_{0} = mem_lhs_{0}[read_base_loc + loc];\n".format(i))

    snippet_file = src_root + 'read_rhs_hbm.h'
    with open(snippet_file, "w") as text_file:
        for i in range(hbm_pc):
            text_file.write("temp_result_{0} = mem_rhs_{0}[read_base_loc + loc];\n".format(i))

    snippet_file = src_root + '../connection.cfg'
    with open(snippet_file, "w") as text_file:
        text_file.write("[connectivity]\nnk=cu_top:1\n")
        # text_file.write("slr=cu_top_1:SLR1\n")
        text_file.write("sp=cu_top_1.mem_indice:HBM[0]\n")
        for i in range(hbm_pc):
            text_file.write("sp=cu_top_1.mem_nnz_{0}:HBM[{1}] \n".format(i, i))
            text_file.write("sp=cu_top_1.mem_col_{0}:HBM[{1}] \n".format(i, hbm_pc+i))
        for i in range(hbm_pc):
            # text_file.write("sp=cu_top_1.mem_lhs_{0}:HBM[{1}] \n".format(i, 2*i))
            # text_file.write("sp=cu_top_1.mem_rhs_{0}:HBM[{1}] \n".format(i, 2*i+1))
            text_file.write("sp=cu_top_1.mem_lhs_{0}:HBM[{1}] \n".format(i, i))
            text_file.write("sp=cu_top_1.mem_rhs_{0}:HBM[{1}] \n".format(i, hbm_pc+i))

    snippet_file = src_root +  'col_interface.h'
    with open(snippet_file, "w") as text_file:
        for i in range(hbm_pc):
            text_file.write("indice_pack_t *mem_col_{},\n".format(i))

    snippet_file = src_root + 'col_func_call.h'
    with open(snippet_file, "w") as text_file:
        for i in range(hbm_pc):
            text_file.write("mem_col_{0},\n".format(i))

    snippet_file = src_root + 'cache_col_hbm_read.h'
    with open(snippet_file, "w") as text_file:
        for i in range(hbm_pc):
            text_file.write("indice_pack_t temp_col_{0} = mem_col_{0}[loc];\n".format(i))

    snippet_file = src_root + 'cache_col_dup_unpack.h'
    with open(snippet_file, "w") as text_file:
        for i in range(hbm_pc):
            text_file.write("duplicate_map[loc][{0}*DATA_PACK_NUM+j] = temp_col_{0}.data[j].range(7, 0);\n".format(i))

    snippet_file = src_root + 'cache_col_loc_unpack.h'
    with open(snippet_file, "w") as text_file:
        for i in range(hbm_pc):
            text_file.write("location_map[loc - duplicate_map_pc_size][{0}*DATA_PACK_NUM+j] = temp_col_{0}.data[j].range(15, 0);\n".format(i))

    snippet_file = src_root + 'mac_tree_stages.h'
    with open(snippet_file, "w") as text_file:
        level_gen_num = int(np.log2(hbm_pc))+1
        for i in reversed(range(level_gen_num)):
            level_num = i+4
            text_file.write("for (int i = 0; i < DATA_PACK_NUM*{}; i++)\n".format(2**i))
            if i == level_gen_num - 1:
                text_file.write("{{node_level_{0}[i]=tmpIn1[i] * tmpIn2[i];}}\n".format(level_num))
            else:
                text_file.write("{{node_level_{0}[i]=node_level_{1}[2*i]+node_level_{1}[2*i+1];}}\n".format(level_num, level_num+1))

            text_file.write("\n")

    snippet_file = src_root + 'pc_load_mem.h'
    with open(snippet_file, "w") as text_file:
        vecbuf_chn_range_bit = int(np.log2(hbm_pc))-1
        text_file.write("#pragma HLS pipeline II={0}\n".format(hbm_pc))
        for i in range(hbm_pc):
            text_file.write("data_pack_t v{0} = mem_data[mem_base_loc + loc + {0}];\n".format(i))

        if vecbuf_chn_range_bit<0:
            text_file.write("u32_t vec_loc = loc;\n".format(i))
        else:
            text_file.write("u32_t vec_loc = loc.range(31,{0});\n".format(vecbuf_chn_range_bit+1))
        for i in range(hbm_pc):
            for j in range(data_pack_num):
                text_file.write("vecbuf[vecbuf_base_loc+vec_loc][{0}]=v{1}.data[{2}];\n".format(i*data_pack_num+j,i,j))

    snippet_file = src_root + 'pc_write_mem.h'
    with open(snippet_file, "w") as text_file:
        vecbuf_chn_range_bit = int(np.log2(hbm_pc))-1
        text_file.write("#pragma HLS pipeline II={0}\n".format(hbm_pc))
        for i in range(hbm_pc):
            text_file.write("data_pack_t v{0};\n".format(i))

        if vecbuf_chn_range_bit<0:
            text_file.write("u32_t vec_loc = loc;\n".format(i))
        else:
            text_file.write("u32_t vec_loc = loc.range(31,{0});\n".format(vecbuf_chn_range_bit+1))
        for i in range(hbm_pc):
            for j in range(data_pack_num):
                text_file.write("v{1}.data[{2}]=vecbuf[vecbuf_base_loc+vec_loc][{0}];\n".format(i*data_pack_num+j,i,j))

        for i in range(hbm_pc):
            text_file.write("mem_data[mem_base_loc + loc + {0}]=v{0};\n".format(i))

    snippet_file = src_root + 'complete_tetris_ii.h'
    with open(snippet_file, "w") as text_file:
        if hbm_pc <=4:
            v_read_ii = 1
        else:
            v_read_ii = 2 

        text_file.write("#pragma HLS pipeline II={0}\n".format(v_read_ii))
        # v_read_partition_num = hbm_pc * data_pack_num//v_read_ii

    snippet_file = src_root + 'fadd_loop_ii.h'
    with open(snippet_file, "w") as text_file:
        text_file.write("#pragma HLS pipeline II={0}\n".format(fadd_ii))

    snippet_file = src_root + 'complete_tetris_unpack_hbm.h'
    with open(snippet_file, "w") as text_file:
        for i in range(hbm_pc):
            text_file.write("bin_to_copy[{0}*DATA_PACK_NUM+j] = temp_col_{0}.data[j].range(7, 0);\n".format(i))

    snippet_file = src_root + 'complete_tetris_hbm_read.h'
    with open(snippet_file, "w") as text_file:
        for i in range(hbm_pc):
            text_file.write("indice_pack_t temp_col_{0} = mem_col_{0}[duplicate_map_offset+loc];\n".format(i))

    snippet_file = src_root + 'tetris_bank_ram_type.h'
    with open(snippet_file, "w") as text_file:
        if ram_type == 'uram':
            """ if not specify will automatically choose bram"""
            text_file.write("#pragma HLS BIND_STORAGE variable=tetris_banks type=RAM_1WNR impl=uram\n")

    snippet_file = src_root + 'mac_tree_nodes_def.h'
    with open(snippet_file, "w") as text_file:
        isca_c = data_pack_num * hbm_pc
        level_gen_num = int(np.log2(isca_c))
        for i in reversed(range(level_gen_num+1)):
            level_num = i
            text_file.write("data_t node_level_{0}[{1}];\n".format(level_num, (2**i)))
            text_file.write("#pragma HLS ARRAY_PARTITION variable=node_level_{0} dim = 0 complete\n".format(level_num))

    snippet_file = src_root + 'mac_tree_interfaces.h'
    with open(snippet_file, "w") as text_file:
        level_gen_num = int(np.log2(isca_c))
        for i in reversed(range(level_gen_num+1)):
            level_num = i
            text_file.write("data_t * node_level_{0}".format(level_num))
            if i == 0:
                text_file.write("\n")
            else:
                text_file.write(",\n")

    snippet_file = src_root + 'mac_tree_func_call.h'
    with open(snippet_file, "w") as text_file:
        level_gen_num = int(np.log2(isca_c))
        for i in reversed(range(level_gen_num+1)):
            level_num = i
            text_file.write("node_level_{0}".format(level_num))
            if i == 0:
                text_file.write("\n")
            else:
                text_file.write(",\n")

    snippet_file = src_root + 'mul_vec_switch.h'
    with open(snippet_file, "w") as text_file:
        acc_pack_width = isca_c//char2nnz(arch_code[0], isca_c)
        for item in range(len(arch_code)):
            case_width = isca_c//char2nnz(arch_code[item], isca_c)
            text_file.write("case {}:\n".format(case_width))

            level_num=int(np.log2(case_width))
            for i in range(case_width):
                text_file.write("\ttemp_v.data[{0}] = node_level_{1}[{0}];\n".format(i, level_num))

            for i in range(case_width, acc_pack_width):
                text_file.write("\ttemp_v.data[{}] = 0;\n".format(i))

            text_file.write("\tbreak;\n")

    snippet_file = src_root + 'align_acc_cnt_switch.h'
    with open(snippet_file, "w") as text_file:
        acc_pack_width = isca_c//char2nnz(arch_code[0], isca_c)
        if len(arch_code)==1:
            text_file.write("align_out[0] << acc_pack.data[0];\n")
        else:
            text_file.write("switch (acc_cnt) {\n")
            for item in range(len(arch_code)):
                case_width = isca_c//char2nnz(arch_code[item], isca_c)
                text_file.write("case {}:\n".format(case_width))
                text_file.write("\tswitch (align_ptr){\n")

                for i in range(acc_pack_width):
                    text_file.write("\tcase {}:\n".format(i))

                    for j in range(case_width):
                        text_file.write("\t\talign_out[{0}] << acc_pack.data[{1}];\n".format((j+i)%acc_pack_width,j))
                    
                    text_file.write("\t\tbreak;\n")

                text_file.write("\t}\n")
                text_file.write("\tbreak;\n")

            text_file.write("}\nalign_ptr += acc_cnt;\n")

    snippet_file = src_root + 'spmv_wb_ii.h'
    with open(snippet_file, "w") as text_file:
        isca_c = hbm_pc * data_pack_num
        acc_pack_width = isca_c//char2nnz(arch_code[0], isca_c)
        text_file.write("#pragma HLS pipeline II={0}\n".format(isca_c//acc_pack_width))

    snippet_file = src_root + 'spmv_acc_pack_num.h'
    with open(snippet_file, "w") as text_file:
        isca_c = hbm_pc * data_pack_num

        acc_pack_width = isca_c//char2nnz(arch_code[0], isca_c)
        text_file.write("#define ACC_PACK_NUM {}\n".format(acc_pack_width))

        spmv_wb_loop2_bound = isca_c//acc_pack_width
        text_file.write("#define SPMV_WB_LOOP2_BOUND {}\n".format(spmv_wb_loop2_bound))

        if acc_pack_width == 1:
            """ in this case the align ptr won't be used so set any bit width other than 0"""
            text_file.write("#define ALIGN_PTR_BITWIDTH 1\n")
        else:
            text_file.write("#define ALIGN_PTR_BITWIDTH {}\n".format(int(np.log2(acc_pack_width))))
    
def main():
    args = parser.parse_args()

    if args.verify == 0:
        csim_verify = False 
    elif args.verify == 1:
        csim_verify = True

    if args.select_bram == 0:
        ram_type = 'uram'
    elif args.select_bram == 1:
        ram_type = 'bram'

    hw_gen(args.arch_code, 
            args.hbm_pc,
            args.max_height,
            args.max_vec_len,
            args.fadd_ii,
            ram_type,
            csim_verify,
            args.output_dir)

if __name__ == '__main__':
    main()