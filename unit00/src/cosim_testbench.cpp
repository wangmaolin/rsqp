#include <iostream>
#include <vector>
#include <fstream>
#include <stdlib.h>
#include <cmath>

#include "constant_type.h"

#define INDICE_MEM_DEPTH 200000
#define DATA_MEM_DEPTH 200000

#define VB_MEM_DEPTH DATA_MEM_DEPTH

#define DATA_WORD_BYTES 4
#define INDICE_WORD_BYTES 4

void verify_results(data_pack_t * mem_truth,
                    data_pack_t * mem_result,
                    data_t diff_eps,
                    int mem_ground_truth_loc,
                    int mem_verify_loc,
                    int vec_pack_len)
{
  int err_cnt = 0;
  int correct_print_cnt = 0;
  data_t max_diff = 0.0;
  int max_diff_loc = -1;

  for(int loc=0; loc<vec_pack_len; loc++)
  {
    data_pack_t truth_temp = mem_truth[mem_ground_truth_loc+loc];
    data_pack_t verify_temp = mem_result[mem_verify_loc+loc];
    for(int k=0; k<DATA_PACK_NUM; k++)
    {
      data_t verify_diff = abs(truth_temp.data[k]-verify_temp.data[k]);

      if(max_diff < verify_diff)
      {
        max_diff = verify_diff;
        max_diff_loc = loc*DATA_PACK_NUM+k;
      }

      if(verify_diff > diff_eps)
      {
        err_cnt +=1;
        if(err_cnt<16)
        {
          std::cout<<"Result mismatch:"
          <<std::left<<std::setw(16)<<loc*DATA_PACK_NUM+k
          <<std::left<<std::setw(16)<<truth_temp.data[k]
          <<std::left<<std::setw(16)<<verify_temp.data[k]
          <<std::endl;
        }
      }
      else{
        correct_print_cnt += 1;
        if(correct_print_cnt<8)
        {
          std::cout<<"CORRECT:"
          <<std::left<<std::setw(16)<<loc*DATA_PACK_NUM+k
          <<std::left<<std::setw(16)<<truth_temp.data[k]
          <<std::left<<std::setw(16)<<verify_temp.data[k]
          <<std::endl;
        }
      }
    }
  }
  std::cout<<"Verify error count:\t"<<err_cnt<<"\tof\t"<<vec_pack_len*DATA_PACK_NUM<<std::endl;
  std::cout<<"Maximum Difference:\t"<<max_diff<<"\tloc\t"<<max_diff_loc<<std::endl;
}

int main()
{

  indice_pack_t* mem_indice = new indice_pack_t[INDICE_MEM_DEPTH];
  #include "tb_array_create.h"

  /* read accelerator program */
  std::ifstream inst_file_stream("cosim.fpga", std::ifstream::binary);

  int meta_info_size = 16;
  std::vector<u32_t> program_info(meta_info_size);
  /* read meta infos about the program */
  std::cout<<"===== Loading Accelerator Program ======"<<std::endl;
  inst_file_stream.read(reinterpret_cast<char*>(program_info.data()), 4*meta_info_size);
  /* check header signature */
  if (program_info[0]!=2135247942)
  {
      std::cout << "WRONG ELF FILE!:" << std::endl;
      return 0;
  }
  std::cout<<"inst_rom pack size: "<< program_info[1] <<std::endl;

  std::cout<<"===== Loading mem indices ======"<<std::endl;
  int indice_mem_bytes = program_info[5] * DATA_PACK_NUM * INDICE_WORD_BYTES;
  std::cout<<"indice mem packs: "<< program_info[5] <<std::endl;
  inst_file_stream.read(reinterpret_cast<char *>(mem_indice), indice_mem_bytes);

  std::cout<<"===== Loading mem col & nnz======"<<std::endl;
  std::cout<<"mem col packs: "<<program_info[13]/DATA_PACK_NUM<<std::endl;
  std::cout<<"mem nnz packs: "<<program_info[14]/DATA_PACK_NUM<<std::endl;
  int col_mem_bytes = program_info[13] * INDICE_WORD_BYTES;
  #include "tb_col_value_copy.h"
  int nnz_mem_bytes = program_info[14] * DATA_WORD_BYTES;
  #include "tb_nnz_value_copy.h"

  std::cout<<"===== Init mem lhs & rhs======"<<std::endl;
  int lr_mem_bytes = program_info[15] * DATA_WORD_BYTES;
  std::cout<<"lhs/rhs mem packs: "<<program_info[15]/DATA_PACK_NUM<<std::endl;
  #include "tb_lr_value_init.h"

  opcode_monitor_stream monitor_stream_out;
  std::cout<<"=====CU START======"<<std::endl;
  /* register file header */
  std::cout<<"ADMM\t";
  std::cout<<"PCG\t";
  std::cout<<"Rho\t";
  std::cout<<"KKT<?eps\t";
  std::cout<<"Prim<?eps\t";
  std::cout<<"Dual<?eps\t";
  std::cout<<std::endl;

  cu_top(mem_indice,
         #include "top_func_call.h"
         program_info[1], 
#ifdef SW_EMU_DEBUG
         0,
         monitor_stream_out);
#else
         0);
#endif

  std::cout<<"op:"<<std::endl;
  int last_opcode = 0;
  while(!monitor_stream_out.empty()) {
     int opcode = monitor_stream_out.read();
     std::cout<<" "<<opcode;

     if(opcode != 1 && last_opcode ==1)
     {
       std::cout << std::endl;
     }
     last_opcode = opcode;
   }
  std::cout<<std::endl;
  /* Debug */
  std::cout<<"===== CU FINISH, VERIFY ";
  int verify_lhs = program_info[9] % 2;
  int mem_verify_loc = program_info[9] >>1;
  if (verify_lhs == 1)
  {
    std::cout<<"Left Loc " <<mem_verify_loc/program_info[10]<<" ====="<<std::endl;
  verify_results(mem_rhs_0, 
                 mem_lhs_0,
                 1e-6,
                 program_info[8], 
                 mem_verify_loc, 
                 program_info[10]);
  }
  else
  {
    std::cout<<"Right Loc " <<mem_verify_loc/program_info[10]<<" ====="<<std::endl;
    verify_results(mem_rhs_0, 
                 mem_rhs_0,
                 1e-6,
                 program_info[8], 
                 mem_verify_loc, 
                 program_info[10]);
  }
  /* Debug Output */

  std::cout<<"ADMM Step: "<<mem_rhs_0->data[16]<<std::endl;
  std::cout<<"Rho      : "<<mem_rhs_0->data[14]<<std::endl;
  std::cout<<"Prim Res : "<<mem_rhs_0->data[17]<<std::endl;
  std::cout<<"Prim Eps: "<<mem_rhs_0->data[18]<<std::endl;
  std::cout<<"Dual Res : "<<mem_rhs_0->data[19]<<std::endl;
  std::cout<<"Dual Eps: "<<mem_rhs_0->data[20]<<std::endl;

  delete [] mem_indice;
  return 0;
}