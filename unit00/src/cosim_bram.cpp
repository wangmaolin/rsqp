#include <iostream>
#include <vector>
#include <fstream>
#include <stdlib.h>
#include <cmath>

#include "constant_type.h"

#define INDICE_MEM_DEPTH 200000
#define DATA_MEM_DEPTH 200000
#define SOL_MEM_DEPTH 10000

#define VB_MEM_DEPTH DATA_MEM_DEPTH

#define DATA_WORD_BYTES 4
#define INDICE_WORD_BYTES 4

void verify_results(data_pack_t * mem_data,
                    data_pack_t * mem_sol,
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
    data_pack_t truth_temp = mem_data[mem_ground_truth_loc+loc];
    data_pack_t verify_temp = mem_sol[mem_verify_loc+loc];
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
  data_pack_t* mem_data = new data_pack_t[DATA_MEM_DEPTH];
  data_pack_t* mem_sol = new data_pack_t[SOL_MEM_DEPTH];
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
  // std::cout<<"indice mem words: "<< program_info[5] * DATA_PACK_NUM <<std::endl;
  std::cout<<"indice mem packs: "<< program_info[5] <<std::endl;
  inst_file_stream.read(reinterpret_cast<char *>(mem_indice), indice_mem_bytes);

  std::cout<<"===== Loading mem data ======"<<std::endl;
  // std::cout<<"mem data words: "<<program_info[7]<<std::endl;
  std::cout<<"data mem packs: "<< program_info[7]/DATA_PACK_NUM <<std::endl;
  inst_file_stream.read(reinterpret_cast<char *>(mem_data), program_info[7]*DATA_WORD_BYTES);

  std::cout<<"===== Loading mem sol======"<<std::endl;
  // std::cout<<"mem sol words: "<<program_info[11]<<std::endl;
  std::cout<<"mem sol packs: "<<program_info[11]/DATA_PACK_NUM<<std::endl;
  inst_file_stream.read(reinterpret_cast<char *>(mem_sol), program_info[11]*DATA_WORD_BYTES);

  std::cout<<"===== Loading mem col & nnz======"<<std::endl;
  std::cout<<"mem col packs: "<<program_info[13]/DATA_PACK_NUM<<std::endl;
  std::cout<<"mem nnz packs: "<<program_info[14]/DATA_PACK_NUM<<std::endl;
  int col_mem_bytes = program_info[13] * INDICE_WORD_BYTES;
  #include "tb_col_value_copy.h"
  int nnz_mem_bytes = program_info[14] * DATA_WORD_BYTES;
  #include "tb_nnz_value_copy.h"

  std::cout<<"===== Init mem lhs & rhs======"<<std::endl;
  #include "tb_value_init.h"

  opcode_monitor_stream monitor_stream_out;
  std::cout<<"=====CU START======"<<std::endl;
  /* register file header */
  std::cout<<"r norm\t";
  std::cout<<"eps|b|\t";
  std::cout<<"alpha\t";
  std::cout<<"beta\t";
  std::cout<<"r.y\t";
  std::cout<<"p.Kp"<<std::endl;

  cu_top(mem_data, 
         mem_indice, 
         mem_sol,
         #include "top_func_call.h"
         program_info[1], 
#ifdef SW_EMU_DEBUG
         0,
         monitor_stream_out);
#else
         0);
#endif

  std::cout<<"op:"<<std::endl;
  while(!monitor_stream_out.empty()) {
     int opcode = monitor_stream_out.read();
     std::cout<<" "<<opcode;

     if(opcode ==1)
     {
       std::cout << std::endl;
     }
   }
  std::cout<<std::endl;

  std::cout<<"=====CU FINISH, VERIFY RESUTS======"<<std::endl;
  verify_results(mem_data, 
                 mem_sol,
                 1e-6,
                 program_info[8], 
                 0, 
                 program_info[10]);

  std::cout<<"PCG iters: "<<mem_sol[program_info[9]].data[9]<<std::endl;

  delete [] mem_indice;
  delete [] mem_data;
  delete [] mem_sol;
  return 0;
}