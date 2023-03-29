void main()
{
	/* ----- declare all the vectors and scalars ----- */
	vectorf work_x, work_x_prev, work_delta_x, work_xtilde_view;
	vectorf work_y, work_delta_y;
	vectorf work_z, work_z_prev, work_ztilde_view;
	vectorf work_data_u, work_data_l, work_data_q;
	vectorf A_mul_x, P_mul_x, At_mul_y;
	matrixf mat_P, mat_A, mat_At;
	vectorf pcg_rhs_part1;
	//variables for update_info
	float max_rel_eps, temp_rel_eps;
	float norm_z, norm_Ax, norm_q, norm_Aty, norm_Px;
	float obj_val, termination, eps_prim, eps_dual;
	float prim_res, dual_res;
	float rho_estimate, prim_res_norm, dual_res_norm;
	float prim_res_div, dual_res_div;
	float admm_steps, rho_update_hint, info_update_hint;

	/* init scalar values */
	float settings_eps_abs = 1e-3; 
	float settings_eps_rel = 1e-3;
	float settings_alpha = 1.6;
	float settings_rho = 0.1;
	float work_rho_inv = 10;
	float work_rho_inv_negative = -10;
	float one_minus_alpha = -0.6; // 1 - alpha 
	float const_plus_1 = 1.0;
	float const_minus_1 = -1.0;
	float settings_sigma = 1e-6;
	float settings_rho_min = 1e-6;
	float settings_rho_max = 1e6;

	float kkt_pcg_epsilon = 1e-8;
	float kkt_pcg_eps_min = 0.5;
	float kkt_pcg_eps_min_prev = 0.5;

	float rho_update_interval = 10.0;
	float rho_update_helper = 8.0; //=rho_update_interval - 2.0

	float solver_max_steps = 4000.0;
	// float solver_max_steps = 0.0;

	/* vectors are init to 0, code start */
	/* swap work_(x/z), work_(x/z)_prev*/
	work_x_prev = work_x;
	work_z_prev = work_z;

	/* ----- update xz_tilde ------ */
	/* compute RHS part1 and part2(stored in ztilde) */
	pcg_rhs_part1 = settings_sigma*work_x_prev - work_data_q;
	/* compiler can't handle -work_rho_inv and I'm too tired to fix ... */
	work_ztilde_view = work_z_prev + work_rho_inv_negative * work_y;
	/* PCG loop has been compiled manually */

	/* update ztilde_view */
	work_ztilde_view = mat_A * work_xtilde_view;

	/* ----- update x -----*/
	work_x = settings_alpha * work_xtilde_view + one_minus_alpha * work_x_prev;
	work_delta_x = work_x - work_x_prev;

	/* ----- update z -----*/
	work_z = settings_alpha * work_ztilde_view + one_minus_alpha * work_z_prev + work_rho_inv * work_y;
	//Note: clip operation
	work_z = work_z > work_data_l;
	work_z = work_z < work_data_u;

	/* ----- update y -----*/
	// the compiler has a bug, -1.0* work_z doesn't work
	work_delta_y = settings_alpha * work_ztilde_view + one_minus_alpha * work_z_prev -  work_z;
	work_delta_y = settings_rho * work_delta_y;
	work_y = work_y + work_delta_y;

	/* ----- compute_rho_estimate ----- */
	rho_update_hint = admm_steps % rho_update_interval;

	// if (rho_update_hint > rho_update_helper || const_plus_1 > admm_steps)
	if (rho_update_hint > rho_update_helper)
	{
		/* update_info */
		/* ----- compute_prim_res ----- */
		A_mul_x = mat_A * work_x;
		work_z_prev = A_mul_x - work_z;
		// skipped condition for scaling settings
		calc_norm_inf(work_z_prev, prim_res);	

		/* ----- compute_dual_res ----- */
		work_x_prev = 1.0 * work_data_q;
		P_mul_x = mat_P * work_x;
		// skipped condition for m
		work_x_prev = work_x_prev + P_mul_x;
		At_mul_y = mat_At * work_y;
		work_x_prev = At_mul_y + work_x_prev;
		// skipped condition for scaling settings
		calc_norm_inf(work_x_prev, dual_res);

		/* ----- compute_prim_tol ----- */
		// skipped condition for scaling settings
		calc_norm_inf(work_z, norm_z);
		calc_norm_inf(A_mul_x, norm_Ax);
		select_max(prim_res_norm, norm_z, norm_Ax);
		eps_prim = settings_eps_rel * prim_res_norm;
		eps_prim = settings_eps_abs + eps_prim;

		/* ----- compute_dual_tol ----- */
		calc_norm_inf(work_data_q, norm_q);
		calc_norm_inf(At_mul_y, norm_Aty);
		select_max(temp_rel_eps, norm_q, norm_Aty);
		calc_norm_inf(P_mul_x, norm_Px);
		select_max(dual_res_norm, temp_rel_eps, norm_Px);
		eps_dual = settings_eps_rel * dual_res_norm;
		eps_dual = settings_eps_abs + eps_dual;

		prim_res_div = prim_res/prim_res_norm;
		dual_res_div = dual_res/dual_res_norm;
		rho_estimate = prim_res_div/dual_res_div;
		c_sqrt(rho_estimate, rho_estimate);

		/* update eps for KKT PCG */
		kkt_pcg_eps_min = prim_res * dual_res;	
		c_sqrt(kkt_pcg_eps_min, kkt_pcg_eps_min);
		kkt_pcg_eps_min = kkt_pcg_eps_min * 0.15;	
		select_min(kkt_pcg_eps_min, kkt_pcg_eps_min_prev, kkt_pcg_eps_min);
		kkt_pcg_eps_min_prev = 1.0 * kkt_pcg_eps_min;
	}

	/* ----- osqp_update_rho ----- */
	if (rho_update_hint > rho_update_helper)
	{
		if (rho_estimate > 2.0 || rho_estimate < 0.5)
		{
			rho_estimate = settings_rho * rho_estimate;
			settings_rho = 1.0 * rho_estimate;
			work_rho_inv = 1.0 / rho_estimate;
			work_rho_inv_negative = const_minus_1 * work_rho_inv;
		}
	}
	/* simple termination check, 
	   add infeasibility check later */
	if(prim_res < eps_prim)
	{
		if(dual_res < eps_dual)
		{
			termination = termination + const_plus_1;
		}
	}

	admm_steps = admm_steps + 1.0;
	if(admm_steps > solver_max_steps)
	{
		termination = termination + const_plus_1;
	}
}