#!/bin/bash

RUN_MODE=$1

if [ -z "$RUN_MODE" ]; then
	echo "please specifiy mode: c(sim), s(ynthesis), r(un), v(iew), or g(ui)"
fi

if [  "$RUN_MODE" = "c" ]; then
	faketime -f "-2y" vitis_hls -f ./tcl/sim.tcl
fi

if [  "$RUN_MODE" = "s" ]; then
	faketime -f "-2y" vitis_hls -f ./tcl/syn.tcl
fi

if [  "$RUN_MODE" = "r" ]; then
	faketime -f "-2y" vitis_hls -f ./tcl/cosim.tcl
fi

if [ "$RUN_MODE" = "v" ] || [ "$RUN_MODE" = "r" ]; then
	# vivado -mode batch -source wcfg.tcl
	# vivado -source wcfg.tcl
	# get total cosimulation time
	cat ./proj_cu_hls/solution1/sim/report/verilog/cu_top.log | grep "PCG iters" | cut -d ' ' -f3 | head -n 1
	cat ./proj_cu_hls/solution1/sim/report/verilog/cu_top.log | grep "finish called" | cut -d ' ' -f6
	cat build_dir.hw.xilinx_u280_xdma_201920_3/proc.link.xclbin.info  | grep Frequency
fi

if [  "$RUN_MODE" = "g" ]; then
	faketime -f "-2y" vitis_hls -p proj_cu_hls
fi