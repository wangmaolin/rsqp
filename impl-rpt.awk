#!/usr/bin/awk -f
BEGIN{
	FS=" +"
	CLB_REG_PRINTED_FLAG=0
	CLB_LUT_PRINTED_FLAG=0
	BRAM_PRINTED_FLAG=0
	URAM_PRINTED_FLAG=0
	DSP_PRINTED_FLAG=0

	KERNEL_FREQ_PRINTED_FLAG=1
	HBM_FREQ_PRINTED_FLAG=1
}
{	
	# for extracting info in impl rpt
	if ($1=="|" && $2=="CLB" && $3=="LUTs"&& CLB_LUT_PRINTED_FLAG==0)
	{
		# printf("LUT\t%s\t%s%%\n",$5, $13)
		printf("%s,%s,",$5, $13)
	CLB_LUT_PRINTED_FLAG=1
	}

	if ($1=="|" && $2=="CLB" && $3=="Registers" && CLB_REG_PRINTED_FLAG==0)
	{
		# printf("REG\t%s\t%s%%\n",$5, $13)
		printf("%s,%s,",$5, $13)
	CLB_REG_PRINTED_FLAG=1
	}

	if ($1=="|" && $2=="Block" && $3=="RAM" &&BRAM_PRINTED_FLAG==0)
	{
		# printf("BRAM\t%s\t%s%%\n",$6, $14)
		printf("%s,%s,",$6, $14)
		BRAM_PRINTED_FLAG=1
	}
	if ($1=="|" && $2=="URAM" && URAM_PRINTED_FLAG==0)
	{
		# printf("URAM\t%s\t%s%%\n",$4, $12)
		printf("%s,%s,",$4, $12)
		URAM_PRINTED_FLAG=1
	}

	if ($1=="|" && $2=="DSPs" && DSP_PRINTED_FLAG==0)
	{
		# printf("DSP\t%s\t%s%%\n",$4, $12)
		printf("%s,%s,",$4, $12)
		DSP_PRINTED_FLAG=1
	}

	# for extracting info in proc.link.xclbin.info
	if ($2=="Name:" && $3=="DATA_CLK")
	{
		KERNEL_FREQ_PRINTED_FLAG=0
	}
	if ($2=="Frequency:" && KERNEL_FREQ_PRINTED_FLAG==0)
	{   
		# printf("Krn-Freq\t%s %s\n",$3, $4)
		printf("%s,",$3)
		KERNEL_FREQ_PRINTED_FLAG=1
	}

	if ($2=="Name:" && $3 ~ /hbm_aclk/)
	{
		HBM_FREQ_PRINTED_FLAG=0
	}
	if ($2=="Frequency:" && HBM_FREQ_PRINTED_FLAG==0)
	{   
		# printf("HBM-Freq\t%s %s\n",$3, $4)
		printf("%s,",$3)
		HBM_FREQ_PRINTED_FLAG=1
	}

	if ($1=="#define" && $2=="TETRIS_HEIGHT" )
	{
		printf("%s,",$3)
	}

	if ($1=="#define" && $2=="UNI_VEC_PACK_SIZE" )
	{
		# this is the last entry, delete the last comma
		printf("%s",$3)
	}
	
}
END{
	# printf("%s,%s,%s,%s,%s,%s,%s,%s,%s,",
	# 		dim_n,
	# 		dim_m,
	# 		nnz,
	# 		iter,
	# 		prim_res,
	# 		dual_res,
	# 		rho_estimate,
	# 		opt,
	# 		time)
}