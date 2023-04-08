#!/bin/bash

for i in "$@"; do
  case $i in
    -c=*|--pc=*)
      HBM_PC="${i#*=}"
      shift # past argument=value
      ;;
    -start=*)
      SCALE_START="${i#*=}"
      shift # past argument=value
      ;;
    -end=*)
      SCALE_BOUND="${i#*=}"
      shift # past argument=value
      ;;
    -v=*|--verify=*)
      SIM_VERIFY="${i#*=}"
      shift # past argument=value
      ;;
    -a=*|--arch=*)
      ARCH_CODE="${i#*=}"
      shift # past argument=value
      ;;
    -b=*)
      BOARD="${i#*=}"
      shift # past argument=value
      ;;
    -g=*)
      GEN_BITFILE=TRUE
      shift # past argument=value
      ;;
    -cvb=*)
      TETRIS_HEIGHT="${i#*=}"
      shift # past argument=value
      ;;
    -vec=*)
      UNI_VEC_PACK="${i#*=}"
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

if [ -z "$SELECT_BRAM" ]; then
	SELECT_BRAM=0
fi

if [ -z "$ADAPT_STRUCT" ]; then
	ADAPT_STRUCT=0
fi

if [ -z "$SIM_VERIFY" ]; then
	SIM_VERIFY=0
fi

if [ -z "$SCALE_BOUND" ]; then
	SCALE_BOUND=19
fi

if [ -z "$SCALE_START" ]; then
	SCALE_START=$SCALE_BOUND
fi

if [ -z "$HBM_PC" ]; then
	HBM_PC=1
fi

if [ -z "$BOARD" ]; then
    BOARD=u50
fi

if [ -z "$ARCH_CODE" ]; then
	case $HBM_PC in
	1)
		ARCH_CODE="e"
		;;

	2)
		ARCH_CODE="f"
		;;

	4)
		ARCH_CODE="g"
		;;
	esac
fi

if [ -z "$FADD_II" ]; then
    FADD_II=4
fi
