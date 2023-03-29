import numpy as np
from inst_set import InstructionSet
from util_opt import compute_precond_diag
from util_opt import data_pack_num
import pandas as pd

from emit_rsqp_ir import compile_update_xzy

def verify_helper(vec_sel_lhs, 
                  vec_addr, 
                  verify_name, 
                  vec_pack_len,
                  isca_c):

    cuda_results = np.zeros(isca_c)

    if verify_name is not None:
        cuda_results = np.fromfile("./verify/"+verify_name, dtype=np.float32)
    mem_verify_loc = ((vec_addr * vec_pack_len)<<1) + vec_sel_lhs

    return cuda_results, mem_verify_loc

def KKT_spmv_seq_gen(cu_isa,
                     cu_dict,
                     p_flag,
                     work_xtilde_view_addr):
    """ instructions to perform a KKT SpMV """
    if p_flag is True:
        """ configuration for Kp0"""
        load_src_sel_lhs = 1
        load_src_addr = 4
    else:
        """ configuration for Kx0"""
        load_src_sel_lhs = 0
        load_src_addr = work_xtilde_view_addr

    """ 4. spmv of Px0"""
    cu_isa.add_load_tetris_inst(src_addr=load_src_addr,
                                src_sel_lhs=load_src_sel_lhs,
                                location_map_offset=cu_dict['offset_P_location_map'],
                                pack_size = cu_isa.sol_vec_pack_len)
    cu_isa.add_complete_tetris_inst(tetris_height=cu_dict['tetris_height_P'],
                                         duplicate_map_offset=cu_dict['offset_P_duplicate_map'])
    cu_isa.add_spmv_inst(mat_id = 0,
                        dst_addr=0,
                        dst_sel_lhs=0)

    """ 5. compute Px0+sigma*I*x0 """
    cu_isa.add_axpby_inst(alpha_addr = 15,
                               beta_addr = 1,
                                src_tetris_addr = 0,
                                src_vf_addr = 0,
                                dst_sel_tetris = 1,
                                dst_addr = 3,
                                pack_size = cu_isa.sol_vec_pack_len)

    """ 8. spmv of Ax0"""
    cu_isa.add_load_tetris_inst(src_addr=load_src_addr,
                                src_sel_lhs=load_src_sel_lhs,
                                location_map_offset=cu_dict['offset_A_location_map'],
                                pack_size = cu_isa.sol_vec_pack_len)
    cu_isa.add_complete_tetris_inst(tetris_height=cu_dict['tetris_height_A'],
                                         duplicate_map_offset=cu_dict['offset_A_duplicate_map'])
    cu_isa.add_spmv_inst(mat_id = 1,
                        dst_addr=0,
                        dst_sel_lhs=0)

    """ 11. spmv of AtAx0"""
    cu_isa.add_load_tetris_inst(src_addr=0,
                                src_sel_lhs=0,
                                location_map_offset=cu_dict['offset_At_location_map'],
                                pack_size = cu_isa.con_vec_pack_len)
    cu_isa.add_complete_tetris_inst(tetris_height=cu_dict['tetris_height_At'],
                                         duplicate_map_offset=cu_dict['offset_At_duplicate_map'])
    cu_isa.add_spmv_inst(mat_id = 2,
                        dst_addr=0,
                        dst_sel_lhs=0)

    """ 12. KKT combine all
        dst = alpha * tetris + beta * vbuf
        the result is at vecbuf[0]
    """
    cu_isa.add_axpby_inst(alpha_addr = 1,
                               beta_addr = 14,
                                src_tetris_addr = 3,
                                src_vf_addr = 0,
                                dst_sel_tetris = 0,
                                dst_addr = 0,
                                pack_size = cu_isa.sol_vec_pack_len)

def program_gen(cu_dict,
                qp_problem,
                hbm_pc):
    """ Init Compiler"""
    cu_isa = InstructionSet(pow(10,8), hbm_pc)
    isca_c = hbm_pc * data_pack_num

    compile_info = {}

    solution_vector_length_padded = cu_dict['solution_vector_length_padded']
    assert(solution_vector_length_padded % isca_c == 0)
    sol_vec_pack_len = solution_vector_length_padded//isca_c
    cu_isa.sol_vec_pack_len = sol_vec_pack_len

    assert(cu_dict['constraint_vector_length_padded'] % isca_c == 0)
    con_vec_pack_len = cu_dict['constraint_vector_length_padded']//isca_c
    cu_isa.con_vec_pack_len = con_vec_pack_len

    unified_vec_pack_len = max(con_vec_pack_len, sol_vec_pack_len)
    cu_isa.unified_vec_pack_len = unified_vec_pack_len

    max_tetris_height = max(cu_dict['location_map'])+1

    """ Compile the other parts of the osqp algorithm """
    cu_dict['sol_vec_pack_len'] = sol_vec_pack_len
    cu_dict['con_vec_pack_len'] = con_vec_pack_len
    cu_dict['unified_vec_pack_len'] = unified_vec_pack_len 

    admm_start_addr = cu_isa.cu_inst_num

    compile_update_xzy(cu_dict)
    update_xzy_insts = np.load('inst-emit.npy')	
    pre_pcg_inst_num = cu_dict['pre_pcg_inst_num']

    """ add the first 4 instructions for compute r1, r2 and swap work_x, work_z"""
    cu_isa.add_external_inst(update_xzy_insts[:4*pre_pcg_inst_num])

    """ Manually compiled PCG instruction sequences 
        0. compute RHS = r1 + rho * At * r2 
        r2 is stored at work_ztilde_view """

    r1_src_addr, r1_src_sel_lhs = cu_dict['pcg_rhs_part1_addr']
    xtilde_src_addr, xtilde_src_sel_lhs = cu_dict['work_xtilde_view_addr']
    r2_src_addr, r2_src_sel_lhs = cu_dict['work_ztilde_view_addr']
    assert xtilde_src_sel_lhs == 0
    assert r1_src_sel_lhs == 1

    cu_isa.add_load_tetris_inst(src_addr=r2_src_addr,
                                src_sel_lhs=r2_src_sel_lhs,
                                location_map_offset=cu_dict['offset_At_location_map'],
                                pack_size = cu_isa.con_vec_pack_len)

    cu_isa.add_complete_tetris_inst(tetris_height=cu_dict['tetris_height_At'],
                                    duplicate_map_offset=cu_dict['offset_At_duplicate_map'])
    """ 0-4 compute At r2"""
    cu_isa.add_spmv_inst(mat_id = 2,
                        dst_addr= 0,
                        dst_sel_lhs=0)

    """ 0-5 compute RHS = r1 + rho * At * r2, note rho and r2 layout """
    cu_isa.add_axpby_inst(alpha_addr = 1, beta_addr = 14,
                                src_tetris_addr = r1_src_addr, src_vf_addr = 0,
                                dst_sel_tetris = 1,
                                dst_addr = 2,
                                pack_size = cu_isa.sol_vec_pack_len)

    """ 1. compute preconditioner on-chip """
    """ 1-4 compute preconditioner M = part1 + rho*part2"""
    cu_isa.add_axpby_inst(alpha_addr = 14, beta_addr = 1,
                            src_tetris_addr = 5, src_vf_addr = 5,
                            dst_sel_tetris = 0,
                            dst_addr = 2,
                            pack_size = sol_vec_pack_len)

    """ 1-5 compute preconditioner inverse of M """
    cu_isa.add_axpby_inst(alpha_addr = 0, beta_addr = 1,
                            src_tetris_addr = 1, src_vf_addr = 2,
                            dst_sel_tetris = 0,
                            dst_addr = 2,
                            pack_size = sol_vec_pack_len,
                            op_type=2)

    """ Compute Kx0 """
    KKT_spmv_seq_gen(cu_isa, cu_dict, p_flag=False, work_xtilde_view_addr = xtilde_src_addr)

    """ 13. compute r0 = -b + Kx0, b at tetris addr 2, Kx0 at vf addr 0"""
    cu_isa.add_axpby_inst(alpha_addr = 2,
                            beta_addr = 1,
                            src_tetris_addr = 2,
                            src_vf_addr = 0,
                            dst_sel_tetris = 1,
                            dst_addr = 1,
                            pack_size = sol_vec_pack_len)

    """ 14. compute y0 = M-1 * r0, op_type 1 means element wise product """
    cu_isa.add_axpby_inst(alpha_addr = 0,
                            beta_addr = 0,
                            src_tetris_addr = 1,
                            src_vf_addr = 2,
                            dst_sel_tetris = 0,
                            dst_addr = 1,
                            pack_size = sol_vec_pack_len,
                            op_type=1)

    """ 15. l2 norm of b """
    """
    cu_isa.add_dot_inst(sel_norm = 1,
                            src_tetris_addr = 2,
                            src_vf_addr = 0 ,
                            dst_reg = 5,
                            pack_size = sol_vec_pack_len)
    """
    cu_isa.add_norm_inf_inst(src_addr=2, 
                             dst_addr=5,
                             src_sel_lhs=1,
                             pack_size=sol_vec_pack_len)

    """ 16. eplison times |b|"""
    cu_isa.add_scalar_op_inst(src_0_reg=3,
                                    src_1_reg=5,
                                    scalar_op=3,
                                    dst_reg=12)

    """	17. r0 dot y0 """
    cu_isa.add_dot_inst(sel_norm = 0,
                            src_tetris_addr = 1,
                            src_vf_addr = 1 ,
                            dst_reg = 7,
                            pack_size = sol_vec_pack_len)

    """ If the norm of r is zero then jump to the end of PCG to avoid NaN """
    """
    cu_isa.add_dot_inst(sel_norm = 1,
                            src_tetris_addr = 1,
                            src_vf_addr = 0 ,
                            dst_reg = 4,
                            pack_size = sol_vec_pack_len)
    """
    cu_isa.add_norm_inf_inst(src_addr=1, 
                             dst_addr=4,
                             src_sel_lhs=1,
                             pack_size=sol_vec_pack_len)

    r_norm_zero_jump_addr = 55
    cu_isa.add_branch_inst(src_0_reg = 13,
                        src_1_reg=4,
                        jump_address=r_norm_zero_jump_addr)

    """ reset pcg_beta """
    cu_isa.add_scalar_op_inst(src_0_reg=0,
                                src_1_reg=0,
                                scalar_op=0,
                                dst_reg=11,
                                imme_flag=0)

    """ PCG init finished, start main pcg loop"""
    pcg_main_loop_program_counter = cu_isa.cu_inst_num

    """ 18. axpby for p(k+1) """
    cu_isa.add_axpby_inst(alpha_addr = 11,
                            beta_addr = 2,
                            src_tetris_addr = 4,
                            src_vf_addr = 1,
                            dst_sel_tetris = 1,
                            dst_addr = 4,
                            pack_size = sol_vec_pack_len)

    """ 19-29. Kp0 spmv """
    KKT_spmv_seq_gen(cu_isa, cu_dict, p_flag=True, work_xtilde_view_addr=xtilde_src_addr)

    """ 30. p dot Kp0 """
    cu_isa.add_dot_inst(sel_norm = 0,
                            src_tetris_addr = 4,
                            src_vf_addr = 0 ,
                            dst_reg = 8,
                            pack_size = sol_vec_pack_len)

    """ 31. compute alpha = dot(r,y)/dot(p,Kp)"""
    cu_isa.add_scalar_op_inst(src_0_reg=7,
                                src_1_reg=8,
                                scalar_op=4,
                                dst_reg=10)


    """ 32. axpby x(k+1) = x(k)+alpha * p(k)"""
    cu_isa.add_axpby_inst(alpha_addr = 10,
                            beta_addr = 1,
                            src_tetris_addr = 4,
                            src_vf_addr = xtilde_src_addr,
                            dst_sel_tetris = 0,
                            dst_addr = xtilde_src_addr,
                            pack_size = sol_vec_pack_len)


    """ 33. axpby r(k+1) = r(k)+alpha * Kp"""
    cu_isa.add_axpby_inst(alpha_addr = 1,
                            beta_addr = 10,
                            src_tetris_addr = 1,
                            src_vf_addr = 0,
                            dst_sel_tetris = 1,
                            dst_addr = 1,
                            pack_size = sol_vec_pack_len)
    """ 34. l2 norm of r1 """
    """
    cu_isa.add_dot_inst(sel_norm = 1,
                            src_tetris_addr = 1,
                            src_vf_addr = 0 ,
                            dst_reg = 4,
                            pack_size = sol_vec_pack_len)
    """
    cu_isa.add_norm_inf_inst(src_addr=1, 
                             dst_addr=4,
                             src_sel_lhs=1,
                             pack_size=sol_vec_pack_len)

    """ 35. compute y(k+1) = M-1 * r(k+1)"""
    cu_isa.add_axpby_inst(alpha_addr = 0,
                            beta_addr = 0,
                            src_tetris_addr = 1,
                            src_vf_addr = 2,
                            dst_sel_tetris = 0,
                            dst_addr = 1,
                            pack_size = sol_vec_pack_len,
                            op_type=1)

    """ 36. move previous r(k) y(k)"""
    cu_isa.add_scalar_op_inst(src_0_reg=7,
                                src_1_reg=0,
                                scalar_op=0,
                                dst_reg=6)
    """ 37. dot r(k+1) y(k+1)"""
    cu_isa.add_dot_inst(sel_norm = 0,
                            src_tetris_addr = 1,
                            src_vf_addr = 1 ,
                            dst_reg = 7,
                            pack_size = sol_vec_pack_len)
    """ 38. compute beta = dot(rk+1,yk+1)/dot(rk,yk)"""
    cu_isa.add_scalar_op_inst(src_0_reg=7,
                                src_1_reg=6,
                                scalar_op=4,
                                dst_reg=11)
    """ pcg iter ++ """
    cu_isa.add_scalar_op_inst(src_0_reg=9,
                                src_1_reg=1,
                                scalar_op=1,
                                dst_reg=9,
                                imme_flag=1)
    """ ===================== End of PCG=================== """

    """ address skipping the following 4 branch instructions"""
    write_back_address = cu_isa.cu_inst_num + 4
    # compare r norm and eps min
    cu_isa.add_branch_inst(src_0_reg = 13,
                        src_1_reg=4,
                        jump_address=write_back_address)

    # compare r norm and eps b norm
    # cu_isa.add_branch_inst(src_0_reg = 12,
    cu_isa.add_branch_inst(src_0_reg = 13,
                        src_1_reg=4,
                        jump_address=write_back_address)

    # compare pcg iters and max iters, using immediate flag
    cu_isa.add_branch_inst(src_0_reg = 9,
                        src_1_reg = 0,
                        jump_address=write_back_address,
                        imme_flag=1,
                        imme_number=19)

    """ uncondition jump back to PCG main loop """
    cu_isa.add_branch_inst(src_0_reg = 1,
                        src_1_reg=0,
                        jump_address=pcg_main_loop_program_counter,
                        imme_flag=1,
                        imme_number=0)

    """ Reset PCG steps """
    cu_isa.add_scalar_op_inst(src_0_reg=0,
                                src_1_reg=0,
                                scalar_op=0,
                                dst_reg=9,
                                imme_flag=0)

    assert cu_dict['post_pcg_jump_offset']==cu_isa.cu_inst_num,\
          print(f"Mismatch {cu_dict['post_pcg_jump_offset']} {cu_isa.cu_inst_num}")
    assert write_back_address == r_norm_zero_jump_addr,\
          print(f"Mismatch {write_back_address } {r_norm_zero_jump_addr}")

    """ read the instructions generated by the emit_ir """
    cu_isa.add_external_inst(update_xzy_insts[4*pre_pcg_inst_num:])

    """ ADMM loop back, exit if reached max_steps """
    admm_end_addr = cu_isa.cu_inst_num + 2

    """ termination in reg 21 """
    cu_isa.add_branch_inst(src_0_reg = 21,
                        src_1_reg = 0,
                        jump_address=admm_end_addr,
                        imme_flag=0,
                        imme_number=0)

    """ Uncondition jump back to ADMM start """
    cu_isa.add_branch_inst(src_0_reg = 1,
                        src_1_reg=0,
                        jump_address=admm_start_addr,
                        imme_flag=1,
                        imme_number=0)

    """ make sure # of inst % 4 == 0"""
    inst_total_num = cu_isa.cu_inst_num
    inst_rom_align_padding = 4 - inst_total_num % 4
    for _ in range(inst_rom_align_padding-1):
        cu_isa.add_inst([cu_isa.inst_halt, 0, 0, 0])
    cu_isa.add_inst([cu_isa.inst_halt, 0, 0, 0], program_end=True)

    assert cu_isa.cu_inst_num < 256, print("Exceed Inst ROM")

    """ reduced P, A, At mat data """
    cu_isa.add_vector_nnz_mem(cu_dict['mat_dat'])
    mem_data_nnz_base_addr = 0
    mem_data_nnz_P_addr = mem_data_nnz_base_addr + cu_dict['offset_P_mat_dat']
    mem_data_nnz_A_addr = mem_data_nnz_base_addr + cu_dict['offset_A_mat_dat']
    mem_data_nnz_At_addr = mem_data_nnz_base_addr + cu_dict['offset_At_mat_dat']

    cu_program_info = np.zeros(16, dtype=np.uint32)
    cu_matrix_info = np.zeros(16, dtype=np.uint32)

    cu_isa.add_vector_col_mem(cu_dict['duplicate_map'])
    cu_isa.add_vector_col_mem(cu_dict['location_map'])
    duplicate_map_pc_size = len(cu_dict['duplicate_map'])//isca_c
    location_map_pc_size = len(cu_dict['location_map'])//isca_c

    """ col idx addr on mem col, the final +4 is due to 4 program rom packs """
    cu_isa.add_vector_col_mem(cu_dict['col_idx'])
    mem_col_base_addr = duplicate_map_pc_size + location_map_pc_size
    mem_col_P_addr = mem_col_base_addr + cu_dict['offset_P_col_idx']
    mem_col_A_addr = mem_col_base_addr + cu_dict['offset_A_col_idx']
    mem_col_At_addr = mem_col_base_addr + cu_dict['offset_At_col_idx']

    cu_program_info[1] = cu_isa.unified_vec_pack_len # program max steps
    cu_program_info[3] = cu_isa.sol_vec_pack_len # sol vec pack len
    cu_program_info[4] = cu_isa.con_vec_pack_len # con vec pack len
    cu_program_info[8] = duplicate_map_pc_size
    cu_program_info[9] = location_map_pc_size + duplicate_map_pc_size

    cu_matrix_info[0] = cu_dict['align_cnt_P']
    cu_matrix_info[1] = cu_dict['col_pack_size_P']
    cu_matrix_info[2] = cu_dict['partial_cnt_P']
    cu_matrix_info[3] = mem_data_nnz_P_addr
    cu_matrix_info[4] = mem_col_P_addr

    cu_matrix_info[5] = cu_dict['align_cnt_A']
    cu_matrix_info[6] = cu_dict['col_pack_size_A']
    cu_matrix_info[7] = cu_dict['partial_cnt_A']
    cu_matrix_info[8] = mem_data_nnz_A_addr
    cu_matrix_info[9] = mem_col_A_addr

    cu_matrix_info[10] = cu_dict['align_cnt_At']
    cu_matrix_info[11] = cu_dict['col_pack_size_At']
    cu_matrix_info[12] = cu_dict['partial_cnt_At']
    cu_matrix_info[13] = mem_data_nnz_At_addr
    cu_matrix_info[14] = mem_col_At_addr

    cu_isa.add_vector_uint32(cu_program_info, record_stage_size=True)
    cu_isa.add_vector_uint32(cu_matrix_info, record_stage_size=True)

    """ Load the register file"""
    cu_register_file = np.load('register_file.npy')	
    rho = cu_register_file[14]
    sigma = cu_register_file[15]
    assert len(cu_register_file) == 64
    rf_pack_size = len(cu_register_file)//data_pack_num
    vector_rf= np.zeros(isca_c*rf_pack_size)
    for i in range(rf_pack_size):
        vector_rf[i*isca_c:i*isca_c + data_pack_num]=cu_register_file[i*data_pack_num:(i+1)*data_pack_num]

    solution_vector_length_original = solution_vector_length_padded - cu_dict['solution_vector_pad_num']

    """ precondition vector m-1 """
    precond_diag = compute_precond_diag(qp_problem, sigma, rho)
    vector_precond_p1_padded = np.zeros(solution_vector_length_padded)
    vector_precond_p1_padded[:solution_vector_length_original]=precond_diag['part1']
    vector_precond_p2_padded = np.zeros(solution_vector_length_padded)
    vector_precond_p2_padded[:solution_vector_length_original]=precond_diag['part2']

    vector_ground_truth = np.zeros(unified_vec_pack_len*isca_c)
    vector_zero_padded = np.zeros(isca_c)
    # vector_random = np.random.rand(solution_vector_length_original)

    """ Skip verification"""
    # cuda_results = vector_zero_padded
    # mem_verify_loc = 9 * unified_vec_pack_len
    
    """ Set verify iter"""
    # reg_layout_df = pd.read_csv('reg_layout.csv')
    # for _, row in reg_layout_df.iterrows():
        # if row['Reg'] == 'solver_max_steps':
            # verify_iter = str(int(row['Value']))

    """ Final solution verification """
    # cuda_results = np.fromfile("./verify/cuda-work_x-"+verify_iter, dtype=np.float32)
    # mem_verify_loc = 9 * unified_vec_pack_len

    """ data_q verify, debug alpha is NaN """
    # cuda_results = np.fromfile("./verify/cuda-work_data_q-"+verify_iter, dtype=np.float32)
    # mem_verify_loc = 6 * unified_vec_pack_len 

    """ linear system solution verification """
    # cuda_results = np.fromfile("./verify/cuda-work_xtilde_view-"+verify_iter, dtype=np.float32)
    # mem_verify_loc = 8 * unified_vec_pack_len

    """ Pre-conditioner"""
    vec_sel_lhs = 1
    vec_addr = 1
    verify_name = None

    """ check the ceil and floor of work_z """
    # cuda_results = np.fromfile("./verify/cuda-work_z-"+verify_iter, dtype=np.float32)
    # mem_verify_loc = 9 * unified_vec_pack_len # LHS

    # cuda_results = np.fromfile("./verify/cuda-work_y-"+verify_iter, dtype=np.float32)
    # mem_verify_loc = 7 * unified_vec_pack_len # LHS

    # cuda_results = np.fromfile("./verify/cuda-work_ztilde_view-"+verify_iter, dtype=np.float32)
    # mem_verify_loc = 8 * unified_vec_pack_len # LHS

    cuda_results, mem_verify_loc = verify_helper(
        vec_sel_lhs,
        vec_addr,
        verify_name,
        vec_pack_len=unified_vec_pack_len,
        isca_c=isca_c)

    vector_ground_truth[:len(cuda_results)] = cuda_results
    mem_ground_truth_loc = 4 * unified_vec_pack_len
    cu_isa.set_verify_info(mem_ground_truth_loc,
                           mem_verify_loc,
                           unified_vec_pack_len)

    """ Init lhs & rhs HBM 
    Read the memory layout CSV file"""
    cu_isa.add_vector_lhs_mem(vector_zero_padded)
    cu_isa.add_vector_lhs_mem(vector_zero_padded)
    cu_isa.add_vector_lhs_mem(vector_zero_padded)
    cu_isa.add_vector_lhs_mem(vector_zero_padded)
    cu_isa.add_vector_lhs_mem(vector_zero_padded)
    cu_isa.add_vector_lhs_mem(vector_precond_p2_padded)

    cu_isa.add_vector_rhs_mem(vector_rf)
    cu_isa.add_vector_rhs_mem(vector_zero_padded)
    cu_isa.add_vector_rhs_mem(vector_zero_padded)
    cu_isa.add_vector_rhs_mem(vector_zero_padded)
    cu_isa.add_vector_rhs_mem(vector_ground_truth)
    cu_isa.add_vector_rhs_mem(vector_precond_p1_padded)

    lr_layout = pd.read_csv('lr_layout.csv')
    vec_init_dict = { 'work_data_l':qp_problem['l'],
                   'work_data_u': qp_problem['u'],
                   'work_data_q': qp_problem['q'],
                #    'work_xtilde_view': vector_random,
    }

    for _, row in lr_layout.iterrows():
        """ Cross check the init book """
        for item in ['LHS', 'RHS']:
            if row[item] in vec_init_dict:
                init_item = vec_init_dict[row[item]]
            else:
                init_item = vector_zero_padded

            if item == 'LHS':
                cu_isa.add_vector_lhs_mem(init_item)
            else:
                cu_isa.add_vector_rhs_mem(init_item)

    assert cu_isa.lhs_mem_words == cu_isa.rhs_mem_words

    compile_info['uni_vec_pack_len']=unified_vec_pack_len
    compile_info['tetris_height']=max_tetris_height
    compile_info['tetris_height_P']=cu_dict['tetris_height_P']
    compile_info['tetris_height_A']=cu_dict['tetris_height_A']
    compile_info['tetris_height_At']=cu_dict['tetris_height_At']
    compile_info['cu_isa'] = cu_isa

    return compile_info
