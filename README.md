# The Reconfigurable Solver for QP
RSQP is a framework for generating FPGA solvers for QPs.
Some off-the-shelf bitstreams are provided to solve generic QPs.
If the sparsity strucutre of QPs to be solved are known, RSQP can analyze the sparsity structure and build a customized FPGA architecture for extra speed boost while solving a series of QPs with the same sparsity pattern.

## System Requirements
This framework requries the following system specifications to run:
- Operating System: Ubuntu 18.04
- FPGA Card: Alevo U50 or U280

Please check this [guide](https://docs.xilinx.com/r/en-US/ug1301-getting-started-guide-alveo-accelerator-cards/Introduction) 
for detailed steps of installing the FPGA card.
To verify the installation, first run the following command to make sure the card is visable:

`xbutil query`

## Part 1. Hello Solver
A pre-synthesised FPGA architecture is in the `bitstream` folder. 
The pre-compiled binary of QP problem to run on this architecture is in the `elf` folder.
Run the following script to see the solver in action 🚀 

`./fpga-solve.sh -sw=svm-s10-2-bf-949-194.fpga -hw=u50-2-bf-3259-275.xclbin`

## Part 2. Compile a QP to Run on an Off-the-shelf Architecture
Generating the elf file:

`./sw-elf-gen.sh -c=2 -a=bf -start=10 -end=10`

To check solver states (values of each vector) in each step, run the functional simulation of the elf:

`./hw-hls-gen.sh -v=1 -c=2 -a=bf -cvb=3259 -vec=275`

`./func-sim.sh -sw=svm-s10-2-bf-949-194.fpga -hw=hbm2-bf-3259-275`

## Part 3. Synthesis a Customzied Architecture for a Specific Sparsity Pattern
To synthesis a new FPGA architecture, Vitis 2021.1 is needed.
Install the tool using this [guide](https://docs.xilinx.com/r/2021.1-English/ug1400-vitis-embedded/Installing-the-Vitis-Software-Platform).

`./hw-hls-gen.sh -c=2 -a=bf -cvb=3259 -vec=275 -b=u50 -g=1`