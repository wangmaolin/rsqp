#!/bin/bash

set -e # Exit on first error

for i in "$@"; do
  case $i in
    -hw=*)
      HW_CFG="${i#*=}"
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

RSQP_ROOT=`pwd`

TOOLCHAIN_DIR=./auto/$HW_CFG
COSIM_ELF=./elf/$ELF_NAME

if [ ! -f $COSIM_ELF ]; then
	echo $COSIM_ELF "DOES NOT EXIST"
	exit 0
fi

if [ ! -d $TOOLCHAIN_DIR ]; then
	echo $TOOLCHAIN_DIR "DOES NOT EXIST"
	exit 0
fi

# Check the compatibility between sw & hw 
HW_HBM_PC=`echo $HW_CFG | cut -d '-' -f1 | sed 's/hbm//'`
HW_ARCH_CODE=`echo $HW_CFG | cut -d '-' -f2`
HW_CVB_HEIGHT=`echo $HW_CFG | cut -d '-' -f3`
HW_VEC_LEN=`echo $HW_CFG | cut -d '-' -f4`

SW_HBM_PC=`echo $ELF_NAME | cut -d '-' -f3`
SW_ARCH_CODE=`echo $ELF_NAME | cut -d '-' -f4`
SW_CVB_HEIGHT=`echo $ELF_NAME | cut -d '-' -f5`
SW_VEC_LEN=`echo $ELF_NAME | cut -d '-' -f6 |sed 's/.fpga//'`

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

echo "----- Copy cosim ELF:$COSIM_ELF -----"
cp $COSIM_ELF ./$TOOLCHAIN_DIR/cosim.fpga

echo "----- Cosim Design in $TOOLCHAIN_DIR -----"
cd $TOOLCHAIN_DIR
./cosim-run.sh c
# ./cosim-run.sh s
# ./cosim-run.sh r

cd $RSQP_ROOT