#!/bin/bash

for i in "$@"; do
  case $i in
    -j=*|--job=*)
      JOB_ID="${i#*=}"
      shift # past argument=value
      ;;
    -*|--*)
      echo "Unknown option $i"
      exit 0
      ;;
    *)
      ;;
  esac
done

# if [ `hostname` != "flatwhite" ]; then
# 	echo "please run on the flatwhite server"
#     exit 0
# fi

if [ -z "$JOB_ID" ]; then
    echo "require job id -j=id"
    exit 0
fi

function job_lock {
	if grep -q "JOB-$JOB_ID" status-job.txt; then
		echo "JOB-$JOB_ID is already running!"
		exit 0
	fi
	echo "JOB-$JOB_ID" >> status-job.txt
}

function job_unlock {
	sed -i "/JOB-$JOB_ID"/d status-job.txt
}

# Run on labpc3 for u50
if [ "$JOB_ID" = "1" ]; then
	job_lock

	./hw-hls-gen.sh -c=4 -a=bg -cvb=409 -vec=51 -b=u50 -g=1
	# ===== 1-be, [[469, 81], [1078, 204], [4749, 550], [41808, 3932]] =====
	# ./hw-hls-gen.sh -c=1 -a=be -cvb=469 -vec=81 -b=u50 -g=1
	# ./hw-hls-gen.sh -c=1 -a=be -cvb=1078 -vec=204 -b=u50 -g=1
	# ./hw-hls-gen.sh -c=1 -a=be -cvb=4749 -vec=550 -b=u50 -g=1
	# ./hw-hls-gen.sh -c=1 -a=be -cvb=41808 -vec=3932 -b=u50 -g=1

	# ===== 1-bce, [[469, 81], [1078, 204], [4749, 550], [41808, 3932]] =====
	# ./hw-hls-gen.sh -c=1 -a=bce -cvb=469 -vec=81 -b=u50 -g=1
	# ./hw-hls-gen.sh -c=1 -a=bce -cvb=1078 -vec=204 -b=u50 -g=1
	# ./hw-hls-gen.sh -c=1 -a=bce -cvb=4749 -vec=550 -b=u50 -g=1
	# ./hw-hls-gen.sh -c=1 -a=bce -cvb=41808 -vec=3932 -b=u50 -g=1

	job_unlock
fi

if [ "$JOB_ID" = "2" ]; then
	job_lock # ------

	# ===== 4-bg, [[166, 21], [409, 51], [1965, 138], [20395, 983]] =====
	./hw-hls-gen.sh -c=4 -a=bg -cvb=166 -vec=21 -b=u50 -g=1
	# ./hw-hls-gen.sh -c=4 -a=bg -cvb=409 -vec=51 -b=u50 -g=1
	# ./hw-hls-gen.sh -c=4 -a=bg -cvb=1965 -vec=138 -b=u50 -g=1
	# ./hw-hls-gen.sh -c=4 -a=bg -cvb=20395 -vec=983 -b=u50 -g=1

	# ===== 4-bcg, [[166, 21], [409, 51], [1965, 138], [20395, 983]] =====
	# ./hw-hls-gen.sh -c=4 -a=bcg -cvb=166 -vec=21 -b=u50 -g=1
	# ./hw-hls-gen.sh -c=4 -a=bcg -cvb=409 -vec=51 -b=u50 -g=1
	# ./hw-hls-gen.sh -c=4 -a=bcg -cvb=1965 -vec=138 -b=u50 -g=1
	# ./hw-hls-gen.sh -c=4 -a=bcg -cvb=20395 -vec=983 -b=u50 -g=1

	job_unlock # ------
fi

if [ "$JOB_ID" = "3" ]; then
	job_lock # ------
	# ===== 2-bf, [[289, 41], [683, 102], [3259, 275], [21344, 1966]] =====
	./hw-hls-gen.sh -c=2 -a=bf -cvb=289 -vec=41 -b=u50 -g=1
	./hw-hls-gen.sh -c=2 -a=bf -cvb=683 -vec=102 -b=u50 -g=1
	./hw-hls-gen.sh -c=2 -a=bf -cvb=3259 -vec=275 -b=u50 -g=1
	./hw-hls-gen.sh -c=2 -a=bf -cvb=21344 -vec=1966 -b=u50 -g=1

	# ===== 2-bcf, [[289, 41], [683, 102], [3259, 275], [21344, 1966]] =====
	./hw-hls-gen.sh -c=2 -a=bcf -cvb=289 -vec=41 -b=u50 -g=1
	./hw-hls-gen.sh -c=2 -a=bcf -cvb=683 -vec=102 -b=u50 -g=1
	./hw-hls-gen.sh -c=2 -a=bcf -cvb=3259 -vec=275 -b=u50 -g=1
	./hw-hls-gen.sh -c=2 -a=bcf -cvb=21344 -vec=1966 -b=u50 -g=1

	job_unlock # ------
fi

if [ "$JOB_ID" = "4" ]; then
	job_lock

	# ===== 4-bcg, [[166, 21], [409, 51], [1965, 138], [20395, 983]] =====
	./hw-hls-gen.sh -c=4 -a=bcg -cvb=166 -vec=21 -b=u280 -g=1
	./hw-hls-gen.sh -c=4 -a=bcg -cvb=409 -vec=51 -b=u280 -g=1
	./hw-hls-gen.sh -c=4 -a=bcg -cvb=1965 -vec=138 -b=u280 -g=1
	./hw-hls-gen.sh -c=4 -a=bcg -cvb=20395 -vec=983 -b=u280 -g=1

	# ===== 2-bcf, [[289, 41], [683, 102], [3259, 275], [21344, 1966]] =====
	./hw-hls-gen.sh -c=2 -a=bcf -cvb=289 -vec=41 -b=u280 -g=1
	./hw-hls-gen.sh -c=2 -a=bcf -cvb=683 -vec=102 -b=u280 -g=1
	./hw-hls-gen.sh -c=2 -a=bcf -cvb=3259 -vec=275 -b=u280 -g=1
	./hw-hls-gen.sh -c=2 -a=bcf -cvb=21344 -vec=1966 -b=u280 -g=1

	job_unlock
fi

if [ "$JOB_ID" = "5" ]; then
	job_lock

	# ===== 4-bg, [[166, 21], [409, 51], [1965, 138], [20395, 983]] =====
	./hw-hls-gen.sh -c=4 -a=bg -cvb=166 -vec=21 -b=u280 -g=1
	./hw-hls-gen.sh -c=4 -a=bg -cvb=409 -vec=51 -b=u280 -g=1
	./hw-hls-gen.sh -c=4 -a=bg -cvb=1965 -vec=138 -b=u280 -g=1
	./hw-hls-gen.sh -c=4 -a=bg -cvb=20395 -vec=983 -b=u280 -g=1

	# ===== 2-bf, [[289, 41], [683, 102], [3259, 275], [21344, 1966]] =====
	./hw-hls-gen.sh -c=2 -a=bf -cvb=289 -vec=41 -b=u280 -g=1
	./hw-hls-gen.sh -c=2 -a=bf -cvb=683 -vec=102 -b=u280 -g=1
	./hw-hls-gen.sh -c=2 -a=bf -cvb=3259 -vec=275 -b=u280 -g=1
	./hw-hls-gen.sh -c=2 -a=bf -cvb=21344 -vec=1966 -b=u280 -g=1

	job_unlock
fi