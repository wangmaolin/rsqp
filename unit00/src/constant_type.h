#pragma once 

#include <ap_int.h>
#include "hls_stream.h"

/* Generated by meta_program.py */
#include "arch_cfg.h"

#define LOC_PACK_SIZE (3*UNI_VEC_PACK_SIZE) 

#define TETRIS_PACK_SIZE TETRIS_HEIGHT
#define REG_FILE_SIZE 64

typedef hls::stream<bool> ctrl_stream;

typedef float data_t;
#define DATA_PACK_NUM 16
/* the Channel C in the ISCA paper */
#define ISCA_C (DATA_PACK_NUM*HBM_PC)

typedef struct data_datatype { data_t data[DATA_PACK_NUM]; } data_pack_t;
typedef hls::stream<data_t> data_stream;

typedef ap_uint<1> u1_t; 
typedef ap_uint<4> u4_t; 
typedef ap_uint<6> u6_t; 
typedef ap_uint<8> u8_t; 
typedef ap_uint<12> u12_t; 
typedef ap_uint<15> u15_t; 
typedef ap_uint<16> u16_t; 
typedef ap_uint<32> u32_t; 

typedef hls::stream<u4_t> opcode_monitor_stream;

typedef struct uint16_datatype { u16_t data[DATA_PACK_NUM]; } uint16_pack_t;

/* indice memory */
typedef ap_uint<32> indice_t; 
typedef struct indice_datatype { indice_t data[DATA_PACK_NUM]; } indice_pack_t;

/* instruction configuration*/
#define INST_FIELD_NUM 4
#define INST_PACK_NUM 4 
typedef ap_uint<32> inst_field_t; 
typedef ap_uint<8> opcode_t; 

/* instruction rom size/4*/
#define INST_PACK_SIZE 64

/* SpMV acc pack siz3*/
#include "spmv_acc_pack_num.h"
typedef struct v_acc_datatype { data_t data[ACC_PACK_NUM]; } spmv_pack_t;
typedef hls::stream<spmv_pack_t> spmv_pack_stream;
typedef hls::stream<u8_t> cnt_pack_stream;
typedef hls::stream<bool> acc_done_stream;
typedef hls::stream<u15_t> fadd_num_stream;
#define CNT_AS_FADD_FLAG 255

void cu_top(indice_pack_t * mem_indice,
            #include "top_interface.h"
            int stage_1_pack_size,
#ifdef SW_EMU_DEBUG
            int skip_cache,
            opcode_monitor_stream & monitor_stream_out);
#else
            int skip_cache);
#endif