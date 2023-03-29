/* cache HBM vector to on-chip vector file and tetris*/
void instruction_load_mem(data_pack_t *mem_data,
                          data_t vecbuf[][ISCA_C],
                          u32_t vecbuf_base_loc,
                          u32_t mem_base_loc,
                          u32_t vec_pack_len)
{
load_data_sol_mem_loop:
    for (u32_t loc = 0; loc < vec_pack_len; loc+=HBM_PC)
    {
        #include "pc_load_mem.h"
    }
}

void instruction_write_mem(data_pack_t *mem_data,
                          data_t vecbuf[][ISCA_C],
                          u32_t vecbuf_base_loc,
                          u32_t mem_base_loc,
                          u32_t vec_pack_len)
{
write_data_sol_mem_loop:
    for (u32_t loc = 0; loc < vec_pack_len; loc+=HBM_PC)
    {
        #include "pc_write_mem.h"
    }
}

u16_t vecbuf_compute_base(u4_t vecbuf_addr)
{
    u16_t vecbuf_base_loc = vecbuf_addr * unified_vec_pac_len;
    return vecbuf_base_loc;
}

/*helper funcion for reading tetris structure*/
void tetris_read(data_t tetris_banks[][ISCA_C],
                 const u16_t location_map[][ISCA_C],
                 u4_t addr_0,
                 u16_t loc,
                 u16_t base_loc,
                 data_t * v)
{
    if(addr_0 == 0)
    {
        tetris_read_loop_1:
        for (int i = 0; i < ISCA_C; i++)
        {
        // #pragma HLS unroll
            // use base_loc to control the offset for matrices P, A, At
            u16_t bank_location = location_map[base_loc + loc][i];
            v[i]=tetris_banks[bank_location][i];
        }
    }
    else{
        tetris_read_loop_2:
        for (int i = 0; i < ISCA_C; i++)
        {
        // #pragma HLS unroll
            v[i] = tetris_banks[base_loc+loc][i];
        }
    }
}

/*helper funcion for writing tetris structure*/
void tetris_write(data_t tetris_banks[][ISCA_C],
                  const u16_t location_map[][ISCA_C],
                  u4_t addr_0,
                  u16_t loc,
                  u16_t base_loc,
                  data_t * v)
{
    if(addr_0 == 0)
    {
        for (int i = 0; i < ISCA_C; i++)
        {
            // use base_loc to control the offset for matrices P, A, At
            u16_t bank_location = location_map[base_loc + loc][i];
            tetris_banks[bank_location][i] = v[i];
        }
    }
    else{
        for (int i = 0; i < ISCA_C; i++)
        {
            tetris_banks[base_loc+loc][i] = v[i];
        }
    }
}

u16_t tetris_compute_base(u4_t src_tetris_addr,
                        u16_t location_map_offset)
{
    u16_t tetris_base_loc;
    if(src_tetris_addr == 0)
    {
        tetris_base_loc = location_map_offset;  
    }
    else
    {
        tetris_base_loc = (src_tetris_addr-1)* unified_vec_pac_len + tetris_vec_offset; 
    }
    return tetris_base_loc;
}

/* alpha*x + beta*y */
void instruction_axpby(data_t tetris_banks[][ISCA_C],
                       const u16_t location_map[][ISCA_C],
                       data_t vecbuf[][ISCA_C],
                       data_t * register_file,
                       u4_t alpha_addr,
                       u4_t beta_addr,
                       u4_t src_tetris_addr,
                       u4_t src_vf_addr,
                       u4_t dst_sel_tetris,
                       u4_t dst_addr,
                       u4_t op_type,
                       // offset for location map of P, A, At
                       u16_t location_map_offset,
                       u16_t vec_pack_len)
{
    /* read alpha and beta from register file*/
    data_t alpha = register_file[alpha_addr];
    data_t beta= register_file[beta_addr];
    /* initial IO base loc setup */
    u16_t tetris_read_base_loc=tetris_compute_base(src_tetris_addr, 
                                                   location_map_offset);  
    u16_t tetris_write_base_loc=tetris_compute_base(dst_addr, 
                                                    location_map_offset);  
    //later need to handle sol_vectors(dim_n) and constrain_vector(dim_m)
    int vecbuf_read_base_loc = vecbuf_compute_base(src_vf_addr);
    int vecbuf_write_base_loc = vecbuf_compute_base(dst_addr);

/* the same location won't be read after update, 
so it's safe to disable loop carry dependency check */
#pragma HLS DEPENDENCE variable=tetris_banks type=INTER direction=RAW dependent=FALSE
#pragma HLS DEPENDENCE variable=vecbuf type=INTER direction=RAW dependent=FALSE

axpby_loop:
        for (int loc = 0; loc < vec_pack_len; loc++)
        {
    #pragma HLS pipeline II = 1
    
            // data_pack_t dst;
            // data_pack_t x;
            data_t dst[ISCA_C];
#pragma HLS ARRAY_PARTITION variable = dst dim = 0 complete
            data_t x[ISCA_C];
#pragma HLS ARRAY_PARTITION variable = x dim = 0 complete

            tetris_read(tetris_banks,
                        location_map,
                        src_tetris_addr,
                        loc,
                        tetris_read_base_loc,
                        x);
                    
            data_t y[ISCA_C];
#pragma HLS ARRAY_PARTITION variable=y dim = 0 complete

            for (int i = 0; i < ISCA_C; i++)
            {
                y[i] = vecbuf[vecbuf_read_base_loc+loc][i];
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
                }
            }

            switch (dst_sel_tetris)
            {
                case 0:
                    for (int i = 0; i < ISCA_C; i++)
                    {
                        vecbuf[vecbuf_write_base_loc+loc][i]=dst[i];
                    }
                    break;
                case 1:
                    tetris_write(tetris_banks,
                                 location_map,
                                 dst_addr,
                                 loc,
                                 tetris_write_base_loc,
                                 dst);
                    break;
            }
        }
}

void spmv_write_to_vbuf(data_t vbuf[][ISCA_C],
                        u32_t result_pack_size,
                        u32_t vbuf_base_write_addr,
                        data_stream align_in[ACC_PACK_NUM])
{
    write_back_loop:
    for (int loc = 0; loc < result_pack_size; loc++)
    {
        #include "spmv_wb_ii.h"
        data_t write_buf[ISCA_C];
#pragma HLS ARRAY_PARTITION variable = write_buf dim = 0 complete

        for (int i = 0; i < SPMV_WB_LOOP2_BOUND; i++)
        {
            for (int j = 0; j < ACC_PACK_NUM; j++)
            {
#pragma HLS unroll
                write_buf[i * ACC_PACK_NUM + j] = align_in[j].read();
            }
        }

        for (int i = 0; i < ISCA_C; i++)
        {
            vbuf[vbuf_base_write_addr + loc][i]=write_buf[i];
        }
    }
}
/* multiply the reduced KKT matrix with a vector*/
void instruction_spmv(data_t tetris_banks[][ISCA_C],
                      #include "top_interface.h"
                      data_t vecbuf[][ISCA_C],
                      u32_t vbuf_base_write_addr,
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
    spmv_write_to_vbuf(vecbuf,
                       result_pack_size,
                       vbuf_base_write_addr,
                       aligned_streams);
}

/* compute dot product (rT, y) & (p, Kp) (r, r) (b , b)*/
void instruction_dot(data_t tetris_banks[][ISCA_C],
                     const u16_t location_map[][ISCA_C],
                     data_t vecbuf[][ISCA_C],
                     data_t * register_file,
                     u4_t sel_norm,
                     u4_t src_tetris_addr,
                     u4_t src_vf_addr,
                     u4_t dst_reg,
                     // offset for location map of P, A, At
                     u16_t location_map_offset,
                     u16_t vec_pack_len)
{
    data_t dot_temp[FADD_SPLIT] = {0.0};
#pragma HLS ARRAY_PARTITION variable = dot_temp dim = 1 complete

    u16_t tetris_read_base_loc=tetris_compute_base(src_tetris_addr, 
                                                   location_map_offset);  
    int vecbuf_read_base_loc = vecbuf_compute_base(src_vf_addr); 
   
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

            tetris_read(tetris_banks,
                        location_map,
                        src_tetris_addr,
                        loc+i,
                        tetris_read_base_loc,
                        x);
            if(sel_norm == 1)
            {
                for (int j = 0; j < ISCA_C; j++)
                {
                    y[j]=x[j];
                }
            }
            else
            { 
                for (int j = 0; j < ISCA_C; j++)
                {
                    y[j] = vecbuf[vecbuf_read_base_loc+loc+i][j];
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

/* the instruction format */
int top_func()
{
    /* vector buffer */
    // data_t vecbuf[VECBUF_PACK_SIZE][ISCA_C];
// #pragma HLS ARRAY_PARTITION variable = vecbuf dim = 2 complete

    /* case 5: BRAM-version
    instruction_spmv(tetris_banks,
                        #include "top_func_call.h"
                        vecbuf,
                        op[2],

                        align_cnt,
                        col_pack_size,
                        partial_cnt,
                        mat_addr,
                        col_addr,
                        result_pack_size);
                        */
/*case 6:
             BRAM-version
            instruction_axpby(tetris_banks,
                              location_map,
                              vecbuf,
                              register_file,
                              op[1].range(3,0),
                              op[1].range(7,4),
                              op[1].range(11,8),
                              op[1].range(15,12),
                              op[1].range(19,16),
                              op[1].range(23,20),
                              op[1].range(27,24),
                              op[2].range(15,0), 
                              op[3]);
            */
        /* case 8:
        BRAM-version
            instruction_dot(tetris_banks,
                            location_map,
                            vecbuf,
                            register_file,
                            op[1].range(3,0),
                            op[1].range(7,4),
                            op[1].range(11,8),
                            op[1].range(15,12),
                            op[2].range(15,0), 
                            sol_vec_pack_len);
                            */
             /* BRAM-version
        case 2:
            instruction_load_mem(mem_data,
                                 vecbuf,
                                 op[1],
                                 op[2],
                                 op[3]);
            break;
            */
            /* BRAM-version
        case 12:
            instruction_load_mem(mem_sol,
                                 vecbuf,
                                 op[1],
                                 op[2],
                                 op[3]);
            break;
                                 */
            /* BRAM-version
        case 14: 
            instruction_write_mem(mem_sol,
                                 vecbuf,
                                 op[1],
                                 op[2],
                                 op[3]);
            break;
                                 */


}