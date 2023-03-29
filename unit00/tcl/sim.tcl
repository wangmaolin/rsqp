# Create a project
open_project -reset proj_cu_hls

# test cu_tetris
add_files src/top_unit.cpp
add_files -tb src/cosim_testbench.cpp

add_files -tb ./cosim.fpga

set_top cu_top

# Create a solution
open_solution -reset solution1 -flow_target vitis

# Define technology and clock rate
set_part  {xcu280-fsvh2892-2L-e}

create_clock -period "300MHz"

csim_design

exit