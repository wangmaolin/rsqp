#!/bin/bash

CUDA_CMD="nvidia-smi | sed '10q;d' |tr -s ' ' | cut -d ' '  -f 5| sed 's/[^0-9]*//g'"

lineNum="$( xbutil query -d 0 | grep -n "Card Power" | head -n 1 | cut -d: -f1)"
POWER_LINE_NUM=$((lineNum+1))
FPGA_CMD="xbutil query -d 0 | sed '"$POWER_LINE_NUM"q;d'"

TIMESTAMP=`date +%s`
# LOG_FILE=./figure/power_trace_$TIMESTAMP.csv

# echo "TIMESTAMP,CUDA_POWER,FPGA_POWER" | tee $LOG_FILE

while true; do
	sleep 0.1

	FPGA_POWER=`eval $FPGA_CMD`
	# CUDA_POWER=`eval $CUDA_CMD`
	TIMESTAMP=`date +%s`
	# echo "$TIMESTAMP, $CUDA_POWER, $FPGA_POWER" >> $LOG_FILE
	echo "$TIMESTAMP, $FPGA_POWER" 
done