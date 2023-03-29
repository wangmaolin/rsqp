void spmv_write_to_hbm(u32_t result_pack_size,
                       #include "mem_lr_interface.h"
                       u6_t dst_addr,
                       u1_t dst_sel_lhs,
                       data_stream align_in[ACC_PACK_NUM])
{
    int write_base_loc = lr_compute_base(dst_addr);

    write_back_loop:
    for (int loc = 0; loc < result_pack_size; loc++)
    {
        #include "spmv_wb_ii.h"
        data_t dst[ISCA_C];
#pragma HLS ARRAY_PARTITION variable = dst dim = 0 complete

        for (int i = 0; i < SPMV_WB_LOOP2_BOUND; i++)
        {
            for (int j = 0; j < ACC_PACK_NUM; j++)
            {
#pragma HLS unroll
                dst[i * ACC_PACK_NUM + j] = align_in[j].read();
            }
        }

        #include "declare_wb_pack.h"
        // pack lr hbm, ISCA_C array to data_pack_t
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

        /* Debug 
        if(loc ==0){
            std::cout << "SpMV results: " << write_base_loc << " "
                      << temp_result_0.data[0] << " "
                      << temp_result_0.data[1] << " "
                      << temp_result_0.data[2] << " "
                      << temp_result_0.data[3] << " "
                      << temp_result_0.data[4] << " "
                      << temp_result_0.data[5] << " "
                      << temp_result_0.data[6] << " "
                      << std::endl;
        }
        */
    }
}

/* multiply the reduced KKT matrix with a vector*/
void instruction_spmv(data_t tetris_banks[][ISCA_C],
                      #include "top_interface.h"
                      u6_t dst_addr,
                      u1_t dst_sel_lhs,
                      u32_t align_cnt,
                      u32_t col_pack_size,
                      u32_t partial_cnt,
                      u32_t mat_addr,
                      u32_t col_addr,
                      u32_t result_pack_size)
{
#pragma HLS DATAFLOW

    fadd_num_stream fadd_num;
#pragma HLS STREAM variable = fadd_num depth = 1024
    data_stream acc_partial;
#pragma HLS STREAM variable = acc_partial depth = 1024
    spmv_pack_stream spmv_pack;
#pragma HLS STREAM variable = spmv_pack depth = 1024
    cnt_pack_stream cnt_pack;
#pragma HLS STREAM variable = cnt_pack depth = 1024
    spmv_mul_vec(tetris_banks, 
                 #include "top_func_call.h"
                 mat_addr,
                 col_addr,
                 col_pack_size,
                 spmv_pack,
                 cnt_pack,
                 fadd_num,
                 acc_partial);

    data_stream acc_complete;
#pragma HLS STREAM variable = acc_complete depth = 1024
    spmv_acc(partial_cnt,
             acc_partial,
             fadd_num,
             acc_complete);

    data_stream aligned_streams[ACC_PACK_NUM];
#pragma HLS STREAM variable = aligned_streams depth = 1024
#pragma HLS ARRAY_PARTITION variable = aligned_streams dim = 1 complete

    spmv_align(align_cnt,
               aligned_streams,
               cnt_pack,
               acc_complete,
               spmv_pack);
/* put the result of mv at vbuf[0] and tetris[3]*/
    spmv_write_to_hbm(result_pack_size,
                      #include "mem_lr_func_call.h"
                      dst_addr,
                      dst_sel_lhs,
                      aligned_streams);
}
