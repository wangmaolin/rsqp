/* dataflow stages for spmv instruction */
void spmv_mul_vec(data_t tetris_banks[][ISCA_C],
                  #include "top_interface.h"
                  u32_t mat_addr,
                  u32_t col_addr,
                  u32_t col_pack_size,
                  spmv_pack_stream &spmv_pack_out,
                  cnt_pack_stream &cnt_pack_out,
                  fadd_num_stream & fadd_num_out,
                  data_stream &acc_partial_out)
{
tetris_mul_loop:
    for (u32_t loc = 0; loc < col_pack_size; loc++)
    {
#pragma HLS pipeline II = 1
        data_t tmpIn1[ISCA_C];
#pragma HLS ARRAY_PARTITION variable = tmpIn1 dim = 0 complete
        data_t tmpIn2[ISCA_C];
#pragma HLS ARRAY_PARTITION variable = tmpIn2 dim = 0 complete
        indice_t tmpCol[ISCA_C];
#pragma HLS ARRAY_PARTITION variable = tmpCol dim = 0 complete

        #include "read_hbm.h"

        for (int j = 0; j < DATA_PACK_NUM; j++)
        {
            #include "unpack_hbm.h"
        }

        u16_t pack_guide = tmpCol[0].range(31, 16);
        /* The LSB indicate if need to accumulate multiple packs */
        bool fadd_flag = pack_guide.range(0, 0);
        u15_t fadd_num = pack_guide.range(15, 1);
        u8_t cnt;
        if (fadd_flag == 1)
        {
        /* means need to do fadd */
            cnt = CNT_AS_FADD_FLAG;
        }
        else{
            cnt = pack_guide.range(8, 1);
        }

        for (int j = 0; j < ISCA_C; j++)
        {
            u16_t tb_col_addr = tmpCol[j].range(15, 0);
            tmpIn2[j]=tetris_banks[tb_col_addr][j];
        }

        #include "mac_tree_nodes_def.h"

        mac_tree(tmpIn1, 
                tmpIn2, 
                #include "mac_tree_func_call.h"
                );

        if(cnt !=0)
        {
            cnt_pack_out << cnt;
        }

        spmv_pack_t temp_v;
        switch (cnt)
        {
        case 0:
            acc_partial_out << node_level_0[0];
            break;

            #include "mul_vec_switch.h"

        /* means need to do fadd */
        case CNT_AS_FADD_FLAG:
            acc_partial_out << node_level_0[0];
            fadd_num_out << fadd_num;
            break;
        }

        if(cnt !=0 && cnt !=CNT_AS_FADD_FLAG)
        {
            spmv_pack_out << temp_v;
        }

    }
}

void spmv_acc(u32_t partial_cnt,
              data_stream &acc_partial_in,
              fadd_num_stream & fadd_num_in,
              data_stream &acc_complete_out)
{
// spmv_acc_loop:
    for (int loc = 0; loc < partial_cnt; loc++)
    {
        u15_t fadd_num = fadd_num_in.read();
        data_t acc_temp[FADD_SPLIT] = {0.0};
#pragma HLS ARRAY_PARTITION variable = acc_temp dim = 1 complete

        ap_uint<FADD_LB> fadd_left_over = fadd_num.range(FADD_LB-1,0);
    acc_split_loop:
        for (int loc = 0; loc < fadd_num; loc+=FADD_SPLIT)
        {
        #include "fadd_loop_ii.h"
            int loop_bound;
            if(loc+FADD_SPLIT > fadd_num)
            {
                loop_bound = fadd_left_over;
            }
            else{
                loop_bound = FADD_SPLIT;
            }

partial_acc_inner_div_loop:
            for(int i = 0; i<FADD_SPLIT;i++)
            {
        #include "fadd_loop_ii.h"
                if(i>=loop_bound)
                {
                    break;
                }
                data_t partial_sum = acc_partial_in.read();
                acc_temp[i] += partial_sum;
            }
        }
 
        for(int i=1; i<FADD_SPLIT;i++)
        {
        #pragma HLS unroll
            acc_temp[0] += acc_temp[i];
        }
        acc_complete_out << acc_temp[0];
    }
}

void spmv_align(int align_cnt,
                data_stream align_out[ACC_PACK_NUM],
                cnt_pack_stream &acc_cnt_in,
                data_stream &acc_complete_in,
                spmv_pack_stream &spmv_pack_in)
{
    ap_uint<ALIGN_PTR_BITWIDTH> align_ptr = 0;
align_loop:
    for (int loc = 0; loc < align_cnt; loc++)
    {
#pragma HLS pipeline II = 1
        u16_t acc_cnt = acc_cnt_in.read();
        spmv_pack_t acc_pack;
        if(acc_cnt == CNT_AS_FADD_FLAG){
            acc_pack.data[0]=acc_complete_in.read();
            acc_cnt = 1;
        }
        else{
            acc_pack = spmv_pack_in.read();
        }
        #include "align_acc_cnt_switch.h"
    }
}
