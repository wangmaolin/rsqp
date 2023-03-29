#!/bin/bash

# Exit on first error
set -e 

source ./arg-env.sh

if [ -z "$TETRIS_HEIGHT" ]; then
	echo "Please specify max tetris height"
	exit 0
fi

if [ -z "$UNI_VEC_PACK" ]; then
	echo "Please specify max vec pack len"
	exit 0
fi

TOOLCHAIN_DIR="./auto/hbm$HBM_PC-$ARCH_CODE-$TETRIS_HEIGHT-$UNI_VEC_PACK"

RSQP_ROOT=`pwd`
echo "RSQP root dir: $RSQP_ROOT"

if [ ! -d "$TOOLCHAIN_DIR" ]; then
	echo "CREATE arch folder +++++ $TOOLCHAIN_DIR +++++" 
	mkdir $TOOLCHAIN_DIR
	mkdir "$TOOLCHAIN_DIR/src"

	ln -sr ./unit00/bitstream-build.sh ./$TOOLCHAIN_DIR/
	ln -sr ./unit00/Makefile ./$TOOLCHAIN_DIR/
	ln -sr ./unit00/utils.mk ./$TOOLCHAIN_DIR/
	ln -sr ./unit00/xrt.ini ./$TOOLCHAIN_DIR/
	ln -sr ./unit00/tcl ./$TOOLCHAIN_DIR/tcl
	ln -sr ./unit00/cosim-run.sh ./$TOOLCHAIN_DIR/

	ln -sr ./unit00/src/cosim_testbench.cpp ./$TOOLCHAIN_DIR/src/
	ln -sr ./unit00/src/host.cpp ./$TOOLCHAIN_DIR/src/
	ln -sr ./unit00/src/top_unit.cpp ./$TOOLCHAIN_DIR/src/
	ln -sr ./unit00/src/constant_type.h ./$TOOLCHAIN_DIR/src/
	ln -sr ./unit00/src/spmv_modules.h ./$TOOLCHAIN_DIR/src/

	ln -sr ./unit00/src/spmv_hbm.h ./$TOOLCHAIN_DIR/src/
	ln -sr ./unit00/src/axpby_hbm.h ./$TOOLCHAIN_DIR/src/
	ln -sr ./unit00/src/dot_hbm.h ./$TOOLCHAIN_DIR/src/
else
	echo "UPDATE arch folder ===== $TOOLCHAIN_DIR ====="
fi

# set max on-chip size
python3 -u meta_program.py\
	--hbm-pc "$HBM_PC"\
	--arch-code "$ARCH_CODE"\
	--select-bram "$SELECT_BRAM"\
	--output-dir "$TOOLCHAIN_DIR"\
	--verify "$SIM_VERIFY"\
	--fadd-ii "$FADD_II"\
	--max-height "$TETRIS_HEIGHT"\
	--max-vec-len "$UNI_VEC_PACK"

if [ ! -z "$GEN_BITFILE" ]; then
	# bitstream build
	echo "----- Build Bitstream in $TOOLCHAIN_DIR -----"
	cd $TOOLCHAIN_DIR
	./bitstream-build.sh -b=$BOARD

	echo "----- Back to RSQP ROOT -----"
	cd $RSQP_ROOT
fi
