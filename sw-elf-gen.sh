#!/bin/bash
# Exit on first error
set -e 

source ./arg-env.sh

for APP_NAME in "Control" "Lasso" "Portfolio" "SVM" "Huber"; do
# for APP_NAME in "Lasso"; do
	for ((SCALE_IDX=$SCALE_START; SCALE_IDX<=$SCALE_BOUND; SCALE_IDX++))
	do
		python3 -u toolchain.py\
			--hbm-pc "$HBM_PC"\
			--arch-code "$ARCH_CODE"\
			--app-name "$APP_NAME"\
			--scale-idx "$SCALE_IDX"\
			--output-dir ./elf
	done
done