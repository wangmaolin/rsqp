from benchmark_gen import problem_instance_gen

from util_opt import mat_pad
from util_opt import data_pack_num
from util_opt import align_columns_with_banks
from util_opt import concat_PAAt_dict
from epec_opt import reduce_ep_ec
from ruiz_eq import struct_adapt
# from ruiz_eq import scale_prob

from link_binary import program_gen

import warnings
warnings.filterwarnings("ignore")
from timeit import default_timer as time
from meta_program import hw_gen

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--hbm-pc', type=int, required=True)
parser.add_argument('--arch-code',type=str, required=True)
parser.add_argument('--output-dir',type=str,  required=True)
parser.add_argument('--app-name',type=str, default='Lasso')
parser.add_argument('--scale-idx', type=int, default=0)


def elf_to_file(app_name, 
                scale_idx, 
                hbm_pc, 
                arch_code, 
                output_dir):

    qp_problem = problem_instance_gen(test_problem_name = app_name, dim_idx = scale_idx)
    isca_c = data_pack_num * hbm_pc

    # scale_prob(qp_problem)

    """ adapt problem structure """
    adapt_struct = False
    if adapt_struct:
        if app_name =='Lasso' or app_name =='Huber' or app_name == 'SVM':
            row_order_At = struct_adapt(qp_problem['A'].transpose(), isca_c)

            P_adapt_row = qp_problem['P'][row_order_At, :]
            P_adapt_full = P_adapt_row[:, row_order_At]

            A_adapt_full=qp_problem['A'][:, row_order_At]

            At_adapt_full =qp_problem['A'].transpose()[row_order_At,:]
        else:
            row_order_A = struct_adapt(qp_problem['A'].tocsr(), isca_c)	
            # At_temp = qp_problem['A'].transpose()[:, row_order_A]
            # row_order_At = struct_adapt(At_temp.tocsr(), isca_c)

            # P_adapt_row = qp_problem['P'][row_order_At, :]
            # P_adapt_full = P_adapt_row[:, row_order_At]
            P_adapt_full = qp_problem['P']

            # A_adapt_row =qp_problem['A'][row_order_A,:]
            # A_adapt_full = A_adapt_row[:, row_order_At]
            A_adapt_full=qp_problem['A'][row_order_A,:]

            # At_adapt_row =qp_problem['A'].transpose()[row_order_At,:]
            # At_adapt_full = At_adapt_row[:, row_order_A]
            At_adapt_full = qp_problem['A'].transpose()[:,row_order_A]
    else:
        P_adapt_full = qp_problem['P']
        A_adapt_full=qp_problem['A']
        At_adapt_full = qp_problem['A'].transpose()

    # P_padded = mat_pad(qp_problem['P'], isca_c)
    P_padded = mat_pad(P_adapt_full, isca_c)
    cu_dict_P = reduce_ep_ec(P_padded, isca_c, arch_code)

    # A_padded = mat_pad(qp_problem['A'], isca_c)
    A_padded = mat_pad(A_adapt_full, isca_c)
    A_aligned = align_columns_with_banks(A_padded, isca_c)
    cu_dict_A = reduce_ep_ec(A_aligned, isca_c, arch_code)

    # At_padded = mat_pad(qp_problem['A'].transpose(), isca_c)
    At_padded = mat_pad(At_adapt_full, isca_c)
    At_aligned = align_columns_with_banks(At_padded, isca_c)
    cu_dict_At = reduce_ep_ec(At_aligned, isca_c, arch_code)

    paat_dict = concat_PAAt_dict(cu_dict_P, 
                                cu_dict_A,
                                cu_dict_At,
                                isca_c)

    compile_info = program_gen(paat_dict, 
                               qp_problem,
                               hbm_pc)

    file_name= output_dir+'/'+app_name.lower().replace(" ", "")\
        +'-s' +str(scale_idx)\
        +'-'+str(hbm_pc)\
        +'-'+arch_code\
        +'-'+str(compile_info['tetris_height'])\
        +'-'+str(compile_info['uni_vec_pack_len'])\
        + '.fpga'

    print('write elf to {}'.format(file_name))
    compile_info['cu_isa'].codegen(file_name)

    return file_name

def main():
    args = parser.parse_args()

    elf_to_file(args.app_name, 
                args.scale_idx,
                args.hbm_pc,
                args.arch_code,
                args.output_dir)

if __name__ == '__main__':
    main()