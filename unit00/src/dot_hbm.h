/* compute dot product (rT, y) & (p, Kp) (r, r) (b , b)*/
void instruction_dot(data_t * register_file,
                     #include "mem_lr_interface.h"
                     u6_t src_lhs_addr,
                     u6_t src_rhs_addr,
                     u6_t dst_reg,
                     u1_t sel_norm,
                     u16_t vec_pack_len)
{
    data_t dot_temp[FADD_SPLIT] = {0.0};
#pragma HLS ARRAY_PARTITION variable = dot_temp dim = 1 complete

    int left_read_base_loc = lr_compute_base(src_lhs_addr);
    int right_read_base_loc = lr_compute_base(src_rhs_addr);
   
    ap_uint<FADD_LB> pack_size_left_over = vec_pack_len.range(FADD_LB-1,0);
inst_dot_loop:
    for (int loc = 0; loc < vec_pack_len; loc+=FADD_SPLIT)
    {
    #include "fadd_loop_ii.h"
        int loop_bound;
        if(loc+FADD_SPLIT > vec_pack_len)
        {
            loop_bound = pack_size_left_over;
        }
        else{
            loop_bound = FADD_SPLIT;
        }
inst_dot_inner_div_loop:
        for(int i = 0; i<FADD_SPLIT;i++)
        {
    #include "fadd_loop_ii.h"
            if(i>=loop_bound)
            {
                break;
            }

            data_t x[ISCA_C];
#pragma HLS ARRAY_PARTITION variable = x dim = 0 complete
            data_t y[ISCA_C];
#pragma HLS ARRAY_PARTITION variable = y dim = 0 complete
            // read lr hbm
            #include "dot_read_lr_hbm.h"
            // unpack lr hbm
            for (int j = 0; j < DATA_PACK_NUM; j++)
            {
                #include "unpack_lr_hbm.h"
            }

            if(sel_norm == 1)
            {
                for (int j = 0; j < ISCA_C; j++)
                {
                    y[j]=x[j];
                }
            }

            #include "mac_tree_nodes_def.h"

            mac_tree(x, 
                     y, 
                     #include "mac_tree_func_call.h"
                     );

            dot_temp[i] += node_level_0[0];
        }
    }

dot_final_acc_loop:    
    for(int i=1; i<FADD_SPLIT;i++)
    {
#pragma HLS unroll
        dot_temp[0] += dot_temp[i];
    }
    
    register_file[dst_reg] = dot_temp[0];
}
