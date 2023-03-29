#include <algorithm>
#include <stdlib.h>
#include <vector>
#include <chrono>
#include <iostream>
#include <fstream>
#include<cmath>
#include "xcl2.hpp"
#include "cmdlineparser.h"
#include "constant_type.h"

void verify_results(data_t * mem_truth,
                    data_t * mem_result,
                    data_t diff_eps,
                    int mem_ground_truth_loc,
                    int mem_verify_loc,
                    int vec_pack_len)
{
  int err_cnt = 0;
  int correct_print_cnt = 0;
  data_t max_diff = 0.0;
  int max_diff_loc = -1;

  for(int loc=0; loc<vec_pack_len*DATA_PACK_NUM; loc++)
  {
    data_t truth_temp = mem_truth[mem_ground_truth_loc*DATA_PACK_NUM+loc];
    data_t verify_temp = mem_result[mem_verify_loc*DATA_PACK_NUM+loc];
    data_t verify_diff = abs(truth_temp-verify_temp);

    if(max_diff < verify_diff)
    {
    max_diff = verify_diff;
    max_diff_loc = loc;
    }

    if(verify_diff > diff_eps)
    {
    err_cnt +=1;
    if(err_cnt<16)
    {
        std::cout<<"Result mismatch:"
        <<std::left<<std::setw(16)<<loc
        <<std::left<<std::setw(16)<<truth_temp
        <<std::left<<std::setw(16)<<verify_temp
        <<std::endl;
    }
    }
    else{
        correct_print_cnt += 1;
        if(correct_print_cnt<8)
        {
            std::cout<<"CORRECT:"
            <<std::left<<std::setw(16)<<loc
            <<std::left<<std::setw(16)<<truth_temp
            <<std::left<<std::setw(16)<<verify_temp
            <<std::endl;
        }
    }
  }
  std::cout<<"Solution diff count:\t"<<err_cnt<<"\tof\t"<<vec_pack_len*DATA_PACK_NUM<<std::endl;
  std::cout<<"Maximum Difference:\t"<<max_diff<<"\tloc\t"<<max_diff_loc<<std::endl;
}

int main(int argc, char** argv) {
    /* Command Line Parser <Full Arg>, <Short Arg>, <Description>, <Default> */
    sda::utils::CmdLineParser parser;
    parser.addSwitch("--xclbin_file", "-x", "input binary file string", "./proc.xclbin");
    parser.addSwitch("--program_name", "-p", "accelerator programm", "./cosim.fpga");
    // parser.addSwitch("--repeat_num", "-r", "repeat number", "1");
    // int repeat_num = parser.value_to_int("repeat_num");
    parser.parse(argc, argv);

    /* read accelerator program */
    std::ifstream inst_file_stream(parser.value("program_name"), std::ifstream::binary);

    int meta_info_size = 16;
    std::vector<u32_t> program_info(meta_info_size);
    /* read meta infos about the program */
    // std::cout<<"===== Loading Accelerator Program ======"<<std::endl;
    inst_file_stream.read(reinterpret_cast<char*>(program_info.data()), 4*meta_info_size);
    /* check header signature */
    if (program_info[0]!=2135247942)
    {
        std::cout << "WRONG BIN FILE!:" << std::endl;
        return EXIT_FAILURE;
    }
    // std::cout<<"inst_rom pack size: "<< program_info[1] <<std::endl;

	cl_int hw_err;
    cl::Context context;
    cl::Program program;
    cl::CommandQueue cmd_queue;
    cl::Kernel cu_krnl;

    std::string binaryFile(parser.value("xclbin_file"));
    auto devices = xcl::get_xil_devices();
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    bool valid_device = false;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        OCL_CHECK(hw_err, context = cl::Context(device, nullptr, nullptr, nullptr, &hw_err));
        OCL_CHECK(hw_err, cmd_queue = cl::CommandQueue(context, device, cl::QueueProperties::Profiling | cl::QueueProperties::OutOfOrder, &hw_err));

        program = cl::Program(context, {device}, bins, nullptr, &hw_err);
        if (hw_err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            // std::cout << "Device[" << i << "]: program successful!\n";
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

    std::string cu_id = std::to_string(1);
    std::string cu_krnls_name_full = std::string("cu_top")+ std::string(":{") + std::string("cu_top_") + cu_id + std::string("}");
    OCL_CHECK(hw_err, cu_krnl = cl::Kernel(program, cu_krnls_name_full.c_str(), &hw_err));

    int indice_mem_words = program_info[5] * DATA_PACK_NUM ;
    std::vector<unsigned int, aligned_allocator<unsigned int>> host_indice_buf(indice_mem_words);
    inst_file_stream.read(reinterpret_cast<char *>(host_indice_buf.data()), indice_mem_words * sizeof(unsigned int));
    cl::Buffer cu_indice_mem;
    OCL_CHECK(hw_err, cu_indice_mem = cl::Buffer(context,
                                                 CL_MEM_USE_HOST_PTR,
                                                 indice_mem_words * sizeof(unsigned int),
                                                 host_indice_buf.data(),
                                                 &hw_err));
    OCL_CHECK(hw_err, hw_err = cu_krnl.setArg(0, cu_indice_mem));
    cmd_queue.enqueueMigrateMemObjects({cu_indice_mem}, 0);// 0 means from host
    cmd_queue.finish();

	int hbm_pc = program_info[12];
	std::vector<cl::Buffer>  cu_col_mem(hbm_pc);
	int col_mem_pc_words = program_info[13];
	std::vector<unsigned int, aligned_allocator<unsigned int>> host_col_buf(col_mem_pc_words);
	for(int i=0; i<hbm_pc;i++)
	{
		inst_file_stream.read(reinterpret_cast<char *>(host_col_buf.data()), col_mem_pc_words*sizeof(unsigned int));

		OCL_CHECK(hw_err, cu_col_mem[i] = cl::Buffer(context,
													 CL_MEM_USE_HOST_PTR,
													 col_mem_pc_words * sizeof(unsigned int),
													 host_col_buf.data(),
													 &hw_err));
		OCL_CHECK(hw_err, hw_err = cu_krnl.setArg(hbm_pc+1+i, cu_col_mem[i]));
		cmd_queue.enqueueMigrateMemObjects({cu_col_mem[i]}, 0 );
		cmd_queue.finish();
	}

	int nnz_mem_pc_words = program_info[14];
	std::vector<cl::Buffer>  cu_nnz_mem(hbm_pc);
	std::vector<float, aligned_allocator<float>> host_nnz_buf(nnz_mem_pc_words);
	for(int i=0; i<hbm_pc;i++) 
	{
		inst_file_stream.read(reinterpret_cast<char *>(host_nnz_buf.data()), nnz_mem_pc_words*sizeof(float));
		OCL_CHECK(hw_err, cu_nnz_mem[i] = cl::Buffer(context,
													CL_MEM_USE_HOST_PTR, 
													nnz_mem_pc_words* sizeof(float),
													host_nnz_buf.data(), 
													&hw_err));
		OCL_CHECK(hw_err, hw_err = cu_krnl.setArg(1+i, cu_nnz_mem[i]));
		cmd_queue.enqueueMigrateMemObjects({cu_nnz_mem[i]}, 0 );
		cmd_queue.finish();
	}

	int lr_mem_pc_words = program_info[15];
	std::vector<float, aligned_allocator<float>> host_lr_buf(lr_mem_pc_words);

	std::vector<cl::Buffer> cu_lhs_mem(hbm_pc);
	for(int i=0; i<hbm_pc;i++) 
	{
        inst_file_stream.read(reinterpret_cast<char *>(host_lr_buf.data()),
                              lr_mem_pc_words * sizeof(float));
        OCL_CHECK(hw_err, cu_lhs_mem[i] = cl::Buffer(context,
                                              CL_MEM_USE_HOST_PTR,
                                              lr_mem_pc_words * sizeof(float),
                                              host_lr_buf.data(),
                                              &hw_err));
        OCL_CHECK(hw_err, hw_err = cu_krnl.setArg(2 * hbm_pc + 1 + i * 2,
                                                  cu_lhs_mem[i]));
        cmd_queue.enqueueMigrateMemObjects({cu_lhs_mem[i]}, 0 );
		cmd_queue.finish();
	}

	std::vector<cl::Buffer> cu_rhs_mem(hbm_pc);
	for(int i=0; i<hbm_pc;i++) 
	{
        inst_file_stream.read(reinterpret_cast<char *>(host_lr_buf.data()),
                              lr_mem_pc_words * sizeof(float));
        OCL_CHECK(hw_err, cu_rhs_mem[i] = cl::Buffer(context,
                                              CL_MEM_USE_HOST_PTR,
                                              lr_mem_pc_words * sizeof(float),
                                              host_lr_buf.data(),
                                              &hw_err));
        OCL_CHECK(hw_err, hw_err = cu_krnl.setArg(2 * hbm_pc + 2 + i * 2,
                                                  cu_rhs_mem[i]));
        cmd_queue.enqueueMigrateMemObjects({cu_rhs_mem[i]}, 0 );
		cmd_queue.finish();
	}

    OCL_CHECK(hw_err, hw_err = cu_krnl.setArg(1 + 4 * hbm_pc,
                                              program_info[1]));
    OCL_CHECK(hw_err, hw_err = cu_krnl.setArg(2 + 4 * hbm_pc,
                                              0));

    //Profiling
    double kernel_time_in_sec = 0;
    std::chrono::duration<double> kernel_time(0);
    auto kernel_start = std::chrono::high_resolution_clock::now();
    cmd_queue.enqueueTask(cu_krnl);
    cmd_queue.finish();
    auto kernel_end = std::chrono::high_resolution_clock::now();
    kernel_time = std::chrono::duration<double>(kernel_end - kernel_start);
    kernel_time_in_sec = kernel_time.count();

    // std::cout << "RSQP finished, FPGA time(ms) -------- "
            //   << kernel_time_in_sec * 1000.0 
            //   <<" --------"
            //   << std::endl;

    /* Copy solution & register file back */
    cmd_queue.enqueueMigrateMemObjects({cu_rhs_mem[0]},
                                       CL_MIGRATE_MEM_OBJECT_HOST);
    cmd_queue.finish();
    /*
    verify_results(host_lr_buf.data(),
                   host_lr_buf.data(),
                   1e-6,
                   program_info[8],
                   program_info[9],
                   program_info[10]);
    */
    std::cout <<"-----"<<parser.value("program_name")<<"-----"<< std::endl;
    // if (std::isnan(host_lr_buf.data()[4]) || std::isnan(host_lr_buf.data()[10]))
    // {
        // std::cout <<"NaN found !!!!"<<std::endl;
    // }
    // else{
    std::cout << std::left << std::setw(10) << "ADMM"
              << std::setw(10) << "Rho"
              << std::setw(10) << "res <?"
              << std::setw(10) << "eps"
              << std::setw(10) << "res <?"
              << std::setw(10) << "eps"
              << std::setw(10) << "res <?"
              << std::setw(10) << "eps"
              << std::setw(10) << "Time(s)" << std::endl;

    std::cout<<std::scientific;
    std::cout.precision(3);
    std::cout << std::left
              << std::setw(10) << static_cast<int>(host_lr_buf.data()[16])
              << std::setw(10) << host_lr_buf.data()[14]

              << std::setw(10) << host_lr_buf.data()[4]
              << std::setw(10) << host_lr_buf.data()[13]

              << std::setw(10) << host_lr_buf.data()[17]
              << std::setw(10) << host_lr_buf.data()[18]

              << std::setw(10) << host_lr_buf.data()[19]
              << std::setw(10) << host_lr_buf.data()[20]
            //   << std::setw(10) << static_cast<int>(kernel_time_in_sec * 1000.0)
              << std::setw(10) << kernel_time_in_sec
              << std::endl;
    // }

    return EXIT_SUCCESS;
}

