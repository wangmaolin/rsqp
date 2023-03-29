#include "constant_type.h"
#include <math.h>

#ifdef REGISTER_DEBUG
#include <list>
#endif

/* global variables*/
u16_t con_vec_pack_len;
u16_t sol_vec_pack_len;
u16_t unified_vec_pac_len;

data_t select_max(data_t a, data_t b)
{
    if (a>b)
    {
        return a;
    }
    else
    {
        return b;
    }
}

data_t select_min(data_t a, data_t b)
{
    if (a<b)
    {
        return a;
    }
    else
    {
        return b;
    }
}

/*helper funcion for 16 input floating point mac*/
void mac_tree(data_t *tmpIn1,
              data_t *tmpIn2,
              #include "mac_tree_interfaces.h"
              )
{
#include "mac_tree_stages.h"

    for (int i = 0; i < 8; i++)
    {node_level_3[i] = node_level_4[2 * i] + node_level_4[2 * i + 1];}

    for (int i = 0; i < 4; i++)
    {node_level_2[i] = node_level_3[2 * i] + node_level_3[2 * i + 1];}

    for (int i = 0; i < 2; i++)
    {node_level_1[i] = node_level_2[2 * i] + node_level_2[2 * i + 1];}

    node_level_0[0] = node_level_1[0] + node_level_1[1];
}

int lr_compute_base(u6_t vecbuf_addr)
{
    int base_loc = vecbuf_addr * unified_vec_pac_len;
    return base_loc;
}

/* instruction spmv sub functions */
#include "spmv_modules.h"
#include "spmv_hbm.h"

/* read all indice mem */
void cache_indice_mem(indice_pack_t *mem_indice,
                      u32_t stage_1_pack_size,
                      u32_t inst_rom[][DATA_PACK_NUM],
                      u32_t * program_info,
                      u32_t * matrix_info)
{
    /* structure to be cached from HBM:
    use int32 with on hbm and int4 and int16 on-chip */
cache_instruction_loop:
    for (int loc = 0; loc < stage_1_pack_size + 2; loc++)
    {
#pragma HLS pipeline II = 1
        indice_pack_t v = mem_indice[loc];
        if (loc < stage_1_pack_size)
        {
            for(int j=0; j<DATA_PACK_NUM; j++)
            {
                inst_rom[loc][j] = v.data[j];
            }
        }
        else if (loc < stage_1_pack_size+1)
        {
            for(int j=0; j<DATA_PACK_NUM; j++)
            {
                program_info[j] = v.data[j];
            }
        }
        else{
            for(int j=0; j<DATA_PACK_NUM; j++)
            {
                matrix_info[j] = v.data[j];
            }
        }
    }
}
/* read location map*/
void cache_col_mem(int duplicate_map_pc_size,
                    int dup_loc_map_pc_size,
                    #include "col_interface.h"
                    u16_t location_map[][ISCA_C])
{

cache_dup_loc_loop : 
for (int loc = duplicate_map_pc_size; loc < dup_loc_map_pc_size; loc++)
    {
#pragma HLS pipeline II = 1
        #include "cache_col_hbm_read.h"
        for(int j=0; j<DATA_PACK_NUM; j++)
        {
            #include "cache_col_loc_unpack.h"
        }
    }
}

void load_reg(data_pack_t *mem_sol,
              u32_t mem_base_loc,
              data_t * register_file) 
{
    for (int loc = 0; loc < 4; loc++)
    {
    data_pack_t v = mem_sol[mem_base_loc+loc];
load_register_loop:
        for (int i = 0; i < DATA_PACK_NUM; i++)
        {
#pragma HLS pipeline II = 8
            register_file[i+loc*DATA_PACK_NUM]=v.data[i];
        }
    }
}

void save_reg(data_pack_t *mem_sol,
              u32_t mem_base_loc,
              data_t * register_file) 
{
    for (int loc = 0; loc < 4; loc++)
    {
        data_pack_t v;
save_register_loop:
        for (int i = 0; i < DATA_PACK_NUM; i++)
        {
#pragma HLS pipeline II = 8
            v.data[i] = register_file[i+loc*DATA_PACK_NUM];
        }
        mem_sol[mem_base_loc+loc]=v;
    }
}

/* compute inf norm of a vector*/
void instruction_norm_inf(data_t *register_file,
                            #include "mem_lr_interface.h"
                            u6_t src_addr,
                            u6_t dst_addr,
                            u1_t src_sel_lhs,
                            u32_t vec_pack_len)
{
    data_t norm_temp[ISCA_C] = {0.0};
#pragma HLS ARRAY_PARTITION variable = norm_temp dim = 0 complete

    int read_base_loc = lr_compute_base(src_addr);
load_tetris_loop:
    for (int loc = 0; loc < vec_pack_len; loc++)
    {
    #pragma HLS pipeline II = 1
        #include "declare_wb_pack.h"
        // read lr hbm
        if (src_sel_lhs){
            #include "read_lhs_hbm.h"
        }
        else{
            #include "read_rhs_hbm.h"
        }

        data_t dst[ISCA_C];
#pragma HLS ARRAY_PARTITION variable = dst dim = 0 complete

        // unpack lr hbm
        for (int j = 0; j < DATA_PACK_NUM; j++)
        {
            #include "unpack_temp_result_hbm.h"
        }

        for (int i = 0; i < ISCA_C; i++)
        {
            data_t element_val = dst[i];
            data_t abs_val = abs(element_val);
            norm_temp[i] = select_max(norm_temp[i], abs_val);
        }

    }

    data_t abs_max = 0.0;
    for (int i = 0; i < ISCA_C; i++)
    {
        if (norm_temp[i]>abs_max)
        {
            abs_max=norm_temp[i];
        }
    }
    register_file[dst_addr] = abs_max;
}

/* load tetris from lhs or rhs memory */
void instruction_load_tetris(data_t tetris_banks[][ISCA_C],
                            #include "mem_lr_interface.h"
                            const u16_t location_map[][ISCA_C],
                            u6_t src_addr,
                            u1_t src_sel_lhs,
                            u32_t location_map_offset,
                            u32_t vec_pack_len)
{
    int read_base_loc = lr_compute_base(src_addr);

load_tetris_loop:
    for (int loc = 0; loc < vec_pack_len; loc++)
    {
    #pragma HLS pipeline II = 1
        #include "declare_wb_pack.h"
        // read lr hbm
        if (src_sel_lhs){
            #include "read_lhs_hbm.h"
        }
        else{
            #include "read_rhs_hbm.h"
        }

        data_t dst[ISCA_C];
#pragma HLS ARRAY_PARTITION variable = dst dim = 0 complete

        // unpack lr hbm
        for (int j = 0; j < DATA_PACK_NUM; j++)
        {
            #include "unpack_temp_result_hbm.h"
        }

        for (int i = 0; i < ISCA_C; i++)
        {
            // use base_loc to control the offset for matrices P, A, At
            u16_t bank_location = location_map[location_map_offset+ loc][i];
            tetris_banks[bank_location][i] = dst[i];
        }
    }
}

/* complete the tetris structure using the duplication guide */
void instruction_complete_tetris(data_t tetris_banks[][ISCA_C],
                                 #include "col_interface.h"
                                 int tetris_height,
                                 u32_t duplicate_map_offset)
{
complete_tetris_loop:
    for (int loc = 0; loc < tetris_height; loc++)
    {
        #include "complete_tetris_ii.h"
        #include "complete_tetris_hbm_read.h"

		u8_t bin_to_copy[ISCA_C];
#pragma HLS ARRAY_PARTITION variable = bin_to_copy dim=0 complete

        for(int j=0; j<DATA_PACK_NUM; j++)
        {
            #include "complete_tetris_unpack_hbm.h"
        }

		data_t v_shift[ISCA_C];
#pragma HLS ARRAY_PARTITION variable = v_shift dim=0 complete 
 		data_t v_write[ISCA_C];
#pragma HLS ARRAY_PARTITION variable = v_write dim=0 complete 
        for (int i = 0; i < ISCA_C; i++)
        {
            v_shift[i] = tetris_banks[loc][i];
            v_write[i] = 0.0;
        }

        for (int s = 0; s < ISCA_C; s ++)
        {
            for (int i = 0; i < ISCA_C; i++)
            {
                if(bin_to_copy[i] == ((i+s)%ISCA_C))
                {
                    v_write[i]=v_shift[i]; 
                }
            }

            data_t v0_temp=v_shift[0];
            for (int i = 0; i < ISCA_C-1; i++)
            {
                v_shift[i]=v_shift[i+1];
            }
            v_shift[ISCA_C-1]=v0_temp;
        }

        for (int i = 0; i < ISCA_C; i++)
        {
            tetris_banks[loc][i] = v_write[i];
        }
    }
}

#include "axpby_hbm.h"
#include "dot_hbm.h"

void instruction_scalar_op(data_t *register_file,
                           u6_t src_0_reg,
                           u6_t src_1_reg,
                           u6_t dst_reg,
                           u4_t scalar_op,
                           u1_t imme_flag)
{
    data_t src0 = register_file[src_0_reg];
    data_t src1;
    /* if flag set, use the reg address as immediate number */
    if (imme_flag == 1) {
        src1 = static_cast<float>(src_1_reg);
    }
    else {
        src1 = register_file[src_1_reg];
    }

    data_t result;
    switch(scalar_op)
    {
        case 0:
            result = src0;
            break;
        case 1:
            result = src0 + src1;
            break;
        case 2:
            result = src0 - src1;
            break;
        case 3:
            result = src0 * src1;
            break;
        case 4:
            result = src0 / src1;
            break;
        case 5:
            result = select_max(src0, src1);
            break;
        case 6:
            result = sqrtf(src0);
            break;
        case 7:
            result = select_min(src0, src1);
            break;
        case 8:
            // result = fmodf(src0, src1);
            result = static_cast<float>(static_cast<int>(src0) % static_cast<int>(src1));
            break;
    }
    register_file[dst_reg] = result;
}

/*jump instruction */
bool instruction_branch(data_t *register_file,
                        u6_t src_0_reg,
                        u6_t src_1_reg,
                        u1_t imme_flag,
                        u16_t imme_number)
{
    data_t src0=register_file[src_0_reg];
    data_t src1;
    /* if flag set, use the reg address as immediate number */
    if (imme_flag == 1) {
        src1 = static_cast<float>(imme_number);
    }
    else {
        src1 = register_file[src_1_reg];
    }

    if (src0 > src1)
    {
        return true;
    }
    else
    {
        return false;
    }
}

void fetch_instruction(u32_t inst_rom[][DATA_PACK_NUM],
                       inst_field_t * op,
                       u32_t program_counter)
{
    u4_t field_ptr = program_counter.range(1, 0) * INST_PACK_NUM;
    for(int j =0; j< INST_FIELD_NUM; j++)
    {
#pragma HLS unroll
        op[j] = inst_rom[program_counter.range(31, 2)][field_ptr + j];
    }
}

void cu_top(indice_pack_t *mem_indice,
            #include "top_interface.h"
            int stage_1_pack_size,
#ifdef SW_EMU_DEBUG
            int skip_cache,
            opcode_monitor_stream & monitor_stream_out)
#else
            int skip_cache)
#endif
{
#pragma HLS INTERFACE m_axi port = mem_indice offset = slave bundle = gmem0 depth = 200000
#include "gmem_bundle.h"

    u32_t inst_rom[INST_PACK_SIZE][DATA_PACK_NUM];
#pragma HLS ARRAY_PARTITION variable = inst_rom dim = 2 complete

    /* tetris data structure */
    data_t tetris_banks[TETRIS_PACK_SIZE][ISCA_C];
#pragma HLS ARRAY_PARTITION variable = tetris_banks dim = 2 complete
    #include "tetris_bank_ram_type.h"

    u16_t location_map[LOC_PACK_SIZE][ISCA_C]; 
#pragma HLS ARRAY_PARTITION variable = location_map dim = 2 complete
#pragma HLS BIND_STORAGE variable=location_map type=RAM_1WNR impl=uram

    data_t register_file[REG_FILE_SIZE];

    u32_t program_info[DATA_PACK_NUM];
#pragma HLS ARRAY_PARTITION variable = program_info dim = 0 complete
    u32_t matrix_info[DATA_PACK_NUM];
#pragma HLS ARRAY_PARTITION variable = matrix_info dim = 0 complete

    /* converge condition */
    u32_t program_counter = 0;

    /* cache instructions and indices */
    if(skip_cache == 0)
    {
        cache_indice_mem(mem_indice, 
                        stage_1_pack_size,
                        inst_rom,
                        program_info,
                        matrix_info);
    }

    // configure global variables
    unified_vec_pac_len = program_info[1];
    sol_vec_pack_len = program_info[3];
    con_vec_pack_len = program_info[4];
    int duplicate_map_pc_size = program_info[8];
    int dup_loc_map_pc_size = program_info[9];

    if(skip_cache == 0)
    {
        cache_col_mem(duplicate_map_pc_size,
                    dup_loc_map_pc_size,
                    #include"col_func_call.h"
                    location_map);
    }

    load_reg(mem_rhs_0, 0, register_file);

    u32_t post_pcg_counter = 1;

    /*spmv configuration*/
    u32_t align_cnt;
    u32_t col_pack_size;
    u32_t partial_cnt;
    u32_t mat_addr;
    u32_t col_addr;
    u32_t result_pack_size;
    int pcg_iters = 0;

    /* For the branch instrution */
    bool branch_flag = false;
    u32_t jump_address = 0;

    /* For the halt instrution */
    bool halt_flag = false;

program_counter_loop:
    while(true)
    {
        /* instruction decoding*/
        inst_field_t op[INST_FIELD_NUM];
        fetch_instruction(inst_rom, op, program_counter);
        opcode_t opcode = op[0].range(3, 0);

        #ifdef SW_EMU_DEBUG
        monitor_stream_out << opcode;
        #endif

        switch (opcode)
        {
        case 1:
            /* branch instruction */
            branch_flag = instruction_branch(register_file,
                               op[1].range(5,0), 
                               op[1].range(11,6),
                               op[1].range(12,12),
                               op[3].range(15,0));
            jump_address = op[2]; 
            break;

        case 2:
            // complete tetris
            instruction_complete_tetris(tetris_banks,
                                        #include"col_func_call.h"
                                        op[1],
                                        op[2]);
            break;
        case 3:
            // multiply reduced kkt
            /* choose which matrix to multiply: P, A, or At*/
            switch(op[1].range(3,0))
            {
                case 0: /*P*/
                    align_cnt = matrix_info[0];
                    col_pack_size = matrix_info[1];
                    partial_cnt = matrix_info[2];
                    mat_addr = matrix_info[3];
                    col_addr = matrix_info[4];
                    result_pack_size = sol_vec_pack_len;
                    break;
                case 1: /*A*/
                    align_cnt = matrix_info[5];
                    col_pack_size = matrix_info[6];
                    partial_cnt = matrix_info[7];
                    mat_addr = matrix_info[8];
                    col_addr = matrix_info[9];
                    result_pack_size = con_vec_pack_len;
                    break;
                case 2: /*At*/
                    align_cnt = matrix_info[10];
                    col_pack_size = matrix_info[11];
                    partial_cnt = matrix_info[12];
                    mat_addr = matrix_info[13];
                    col_addr = matrix_info[14];
                    result_pack_size = sol_vec_pack_len;
                    break;
            }
            instruction_spmv(tetris_banks,
                             #include "top_func_call.h"
                             op[2].range(5, 0),
                             op[2].range(6, 6),
                             align_cnt,
                             col_pack_size,
                             partial_cnt,
                             mat_addr,
                             col_addr,
                             result_pack_size);

            break;
        case 4:
            instruction_axpby(register_file,
                             #include "mem_lr_func_call.h"
                              op[1].range(5, 0),
                              op[1].range(11, 6),
                              op[1].range(17, 12),
                              op[1].range(23, 18),
                              op[1].range(29, 24),
                              op[1].range(30, 30),
                              op[2].range(3, 0),
                              op[3]);

            break;
        case 5:
            instruction_load_tetris(tetris_banks,
                             #include "mem_lr_func_call.h"
                             location_map,
                              op[1].range(5, 0),
                              op[1].range(6, 6),
                              op[2],
                              op[3]);
            break;
        case 6:
           instruction_dot(register_file,
                            #include "mem_lr_func_call.h"
                            op[1].range(5,0),
                            op[1].range(11,6),
                            op[1].range(17,12),
                            op[1].range(18,18),
                            sol_vec_pack_len);

            break;
        case 7:
            instruction_scalar_op(register_file,
                                  op[1].range(5,0),
                                  op[1].range(11,6),
                                  op[1].range(17,12),
                                  op[1].range(21,18),
                                  op[1].range(22,22));
            break;
        case 8:
           instruction_norm_inf(register_file,
                            #include "mem_lr_func_call.h"
                            op[1].range(5,0),
                            op[1].range(11,6),
                            op[1].range(12,12),
                            op[2]);
            break;
        default:
            /* halt */
            halt_flag = true;
            break;
        }

        /* ====== Debug ====== 
        std::cout.precision(3);
        std::cout<<"Op-";
        std::cout<<opcode<<"\t";
        std::list<int> reg_debug_list = {16, 9, 4, 5, 6, 7, 8};

        for (auto it = reg_debug_list.begin(); it != reg_debug_list.end(); ++it) {
            std::cout<<"R-"<< *it <<" "<<register_file[*it]<<"\t";
        }
        std::cout<<std::endl;

        int nan_flag = 0;
        for (int i=0; i<64; i++)
        {
            if (isnan(register_file[i])){
                std::cout<<"R-"<<i<<" is NaN!"<<std::endl;
                nan_flag = 1;
            }
        }
        if(nan_flag == 1)
        {
            break;
        }
        /* ====== Debug ====== */
 
        if(halt_flag)
        {
           save_reg(mem_rhs_0, 0, register_file);
            /* exit main loop */
            break;
        }

        if(branch_flag)
        {
            #ifdef REGISTER_DEBUG
            std::cout.precision(3);
            std::cout<<register_file[16]<<"\t";
            std::cout<<register_file[9]<<"\t";
            std::cout<<register_file[14]<<"\t";

            std::cout<<register_file[4]<<" <? ";
            std::cout<<register_file[13]<<"\t";
            std::cout<<register_file[17]<<" <? ";
            std::cout<<register_file[18]<<"\t";
            std::cout<<register_file[19]<<" <? ";
            std::cout<<register_file[20]<<"\t";
            std::cout<<std::endl;
            #endif
 
            program_counter = jump_address;
            branch_flag = false;
        }
        else{
            program_counter = program_counter + 1;
        }
    }
}
