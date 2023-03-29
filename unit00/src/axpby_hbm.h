void instruction_axpby(data_t * register_file,
                        #include "mem_lr_interface.h"
                       u6_t alpha_addr,
                       u6_t beta_addr,
                       u6_t src_lhs_addr,
                       u6_t src_rhs_addr,
                       u6_t dst_addr,
                       u1_t dst_sel_lhs,
                       u4_t op_type,
                       u16_t vec_pack_len)
{
    /* read alpha and beta from register file*/
    data_t alpha = register_file[alpha_addr];
    data_t beta = register_file[beta_addr];
    /* initial IO base loc setup */
    int left_read_base_loc = lr_compute_base(src_lhs_addr);
    int right_read_base_loc = lr_compute_base(src_rhs_addr);
    int write_base_loc = lr_compute_base(dst_addr);

    #include "lr_depend_disable.h"

axpby_loop:
        for (int loc = 0; loc < vec_pack_len; loc++)
        {
    #pragma HLS pipeline II = 1

            data_t dst[ISCA_C];
#pragma HLS ARRAY_PARTITION variable = dst dim = 0 complete
            data_t x[ISCA_C];
#pragma HLS ARRAY_PARTITION variable = x dim = 0 complete
            data_t y[ISCA_C];
#pragma HLS ARRAY_PARTITION variable=y dim = 0 complete
            // read lr hbm
            #include "read_lr_hbm.h"
            #include "declare_wb_pack.h"
            // unpack lr hbm
            for (int j = 0; j < DATA_PACK_NUM; j++)
            {
                #include "unpack_lr_hbm.h"
            }

            for (int i = 0; i < ISCA_C; i++)
            {
                switch(op_type) 
                {   case 0:
                        dst[i] = alpha*x[i] + beta * y[i];
                        break;
                    case 1:
                        dst[i] = x[i] * y[i];
                        break;
                    case 2:
                        if (y[i] == 0.0)
                        {
                            dst[i] = 0.0; 
                        }
                        else{
                            dst[i] = 1.0 / y[i];
                        }
                        break;
                    case 3:
                        dst[i]= select_min(x[i],y[i]);
                        break;
                    case 4:
                        dst[i]= select_max(x[i],y[i]);
                        break;
                }
            }

            // pack lr hbm
            for (int j = 0; j < DATA_PACK_NUM; j++)
            {
                #include "pack_lr_hbm.h"
            }

            // write lr hbm 
            if (dst_sel_lhs){
                #include "write_lhs_hbm.h"
            }
            else{
                #include "write_rhs_hbm.h"
            }
        }
}
