#!/bin/sh
for i in "$@"; do
  case $i in
    -b=*|--board=*)
      BOARD="${i#*=}"
      shift # past argument=value
      ;;
    -t=*|--target=*)
      BUILD_TARGET="${i#*=}"
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
if [ -z "$BOARD" ]; then
    TARGET_DEVICE=xilinx_u280_xdma_201920_3
    # TARGET_DEVICE=xilinx_u50_gen3x16_xdma_201920_3
fi

if [ "$BOARD" = "u280" ]; then
    TARGET_DEVICE=xilinx_u280_xdma_201920_3
else
    TARGET_DEVICE=xilinx_u50_gen3x16_xdma_201920_3
fi

if [ -z "$BUILD_TARGET" ]; then
    BUILD_ALL=all
    BUILD_TARGET=hw
else
    BUILD_ALL=host
    BUILD_TARGET=hw
fi

faketime -f "-2y" make $BUILD_ALL TARGET=$BUILD_TARGET DEVICE=$TARGET_DEVICE
