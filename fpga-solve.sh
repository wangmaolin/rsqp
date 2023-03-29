#!/bin/bash

set -e # Exit on first error

for i in "$@"; do
  case $i in
    -hw=*)
      BIT_STREAM="${i#*=}"
      shift # past argument=value
      ;;
    -sw=*)
      ELF_NAME="${i#*=}"
      shift # past argument=value
      ;;
    -*|--*)
      echo "Unknown option $i"
      exit 1
      ;;
    *)
      ;;
  esac
done

ELF_PATH=./elf
BIT_PATH=./bitstream

if [ ! -f $BIT_PATH/$BIT_STREAM ] || [ ! -f $ELF_PATH/$ELF_NAME ]; then
	echo $ELF_NAME "OR" $BIT_STREAM "DOES NOT EXIST"
	exit 0
fi

# Check the compatibility between sw & hw 
HW_HBM_PC=`echo $BIT_STREAM | cut -d '-' -f2`
HW_ARCH_CODE=`echo $BIT_STREAM | cut -d '-' -f3`
HW_CVB_HEIGHT=`echo $BIT_STREAM | cut -d '-' -f4`
HW_VEC_LEN=`echo $BIT_STREAM | cut -d '-' -f5 | sed 's/.xclbin//'`

SW_HBM_PC=`echo $ELF_NAME | cut -d '-' -f3`
SW_ARCH_CODE=`echo $ELF_NAME | cut -d '-' -f4`
SW_CVB_HEIGHT=`echo $ELF_NAME | cut -d '-' -f5`
SW_VEC_LEN=`echo $ELF_NAME | cut -d '-' -f6 | sed 's/.fpga//'`

if [[ $HW_HBM_PC != $SW_HBM_PC ]] ||\
   [[ $HW_ARCH_CODE != $SW_ARCH_CODE ]]; then
	echo $HW_HBM_PC $HW_ARCH_CODE "DON't MATCH"\
	     $SW_HBM_PC $SW_ARCH_CODE
	exit 0
fi

if [ "$SW_CVB_HEIGHT" -gt "$HW_CVB_HEIGHT" ] ||\
   [ "$SW_VEC_LEN" -gt "$HW_VEC_LEN" ]; then
	echo $SW_CVB_HEIGHT $SW_VEC_LEN "EXCEED"\
	     $HW_CVB_HEIGHT $HW_VEC_LEN
	exit 0
fi

./solver/rsqp -x $BIT_PATH/$BIT_STREAM -p $ELF_PATH/$ELF_NAME
