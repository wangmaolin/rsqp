import pandas as pd
from pycparser import parse_file
from pycparser import c_ast
import numpy as np
from inst_set import InstructionSet

def get_Decl_info(node):
    if node.init is not None:
        init_flag = 'yes'
        if hasattr(node.init, 'value'):
            init_value = float(node.init.value)
        else:
            """ init negative values """
            init_value = -float(node.init.expr.value)
    else:
        init_flag = 'unset'
        init_value= 0.0

    id_type_record =[node.name, node.type.type.names[0], node.coord, init_flag, init_value]
    return id_type_record

def lookup_var_type(df, var):
    assert hasattr(var, 'name')
    id=var.name
    id_record = df.loc[df['id'] == id]
    """ the ID should be unique """
    assert len(id_record) == 1
    id_type = id_record.iloc[0]['type']
    return id_type

def df_insert_row(df, row):
    df.loc[len(df)] = row

def cls_name(node):
    return node.__class__.__name__

class EmitVisitor(c_ast.NodeVisitor):
    def __init__(self, cu_dict):
        self.cu_dict = cu_dict
        self.symbol_table = pd.DataFrame(columns=['id','type','src', 'init_flag', 'const_value'])

        """ use the tuple of operand and id type as the key to rule dict """
        self.binop_prod_rule={
            ('+', 'vectorf', 'vectorf') : ('axpby','linear'),
            ('-', 'vectorf', 'vectorf') : ('axpby','linear'),
            ('<', 'vectorf', 'vectorf') : ('axpby','select_min'),
            ('>', 'vectorf', 'vectorf') : ('axpby','select_max'),
            ('*', 'vectorf', 'vectorf') : ('dot', None),

            ('*', 'float', 'vectorf') : ('axpby_frac', None),

            ('*', 'matrixf', 'vectorf') : ('spmv', None),

            ('+', 'float', 'float') : ('scalar_op', 'add'),
            ('-', 'float', 'float') : ('scalar_op', 'sub'),
            ('*', 'float', 'float') : ('scalar_op', 'mul'),
            ('/', 'float', 'float') : ('scalar_op', 'div'),
            ('%', 'float', 'float') : ('scalar_op', 'mod'),

            ('>', 'float', 'float') : ('branch', 'gt'),
            ('<', 'float', 'float') : ('branch', 'lt'),

            ('||', 'boolean', 'boolean') : ('branch', 'OR'),
            ('&&', 'boolean', 'boolean') : ('branch', 'AND'),
        }

        self.inst_out_type={
            'axpby': 'vectorf',
            'dot':'float',
            'axpby_frac': 'vectorf',
            'spmv':'vectorf',
            'scalar_op':'float',
            'branch':'boolean',
        }

        self.axpby_type_dict ={'linear':0, 'select_min':3, 'select_max': 4}  
        self.mat_id_dict ={'mat_P':(0, 'sol_vec_pack_len'), 
                           'mat_A':(1, 'sol_vec_pack_len'),
                           'mat_At': (2,'con_vec_pack_len')}  

        self.scalar_op_type_dict ={'add':1, 
                                   'sub':2, 
                                   'mul': 3,
                                   'div': 4,
                                   'select_max': 5,
                                   'c_sqrt': 6,
                                   'select_min': 7,
                                   'mod':8,
                                   }  

        """ binary op temp var counter """ 
        self.temp_var_idx=0
        self.const_var_idx=0

        """ buffer to hold fractions of axpby""" 
        self.axpby_buffer={'s_a': None,
                            'v_x':None,
                            's_b':None,
                            'v_y':None,
                            'frac_num':0}

        """ IR table"""
        self.ir_table = pd.DataFrame(columns=['inst_type', 'op_type','result',
                                        'v0','v1','s0','s1'])

        """ Two way partition for vector input of axpby instructions """
        self.bp_graph_nodes = []
        self.bp_graph_edges = [] 

        """ vector memory layout """
        self.lhs_or_rhs_hbm = [] # can be either lhs or rhs
        self.lhs_hbm = []
        self.rhs_hbm = []
        self.lr_layout = pd.DataFrame(columns=['LHS', 'RHS'])
        self.vec_addr_offset = 6

        """ register file layout with pre-defined map """
        self.reg_onchip = list(map(lambda x:'pcg-reg-'+str(x), range(16)))
        self.reg_onchip += ['admm_steps', 
                            'prim_res', 
                            'eps_prim',
                            'dual_res', 
                            'eps_dual',
                            'termination']
        self.reg_layout = pd.DataFrame(columns=['Reg', 'Value'])
        """ Old register layout from PCG's manual compilation"""
        self.reg_onchip[1] = 'const_plus_1'
        self.reg_onchip[2] = 'const_minus_1'
        self.reg_onchip[3] = 'kkt_pcg_epsilon'
        self.reg_onchip[13] = 'kkt_pcg_eps_min'
        self.reg_onchip[14] = 'settings_rho'
        self.reg_onchip[15] = 'settings_sigma'

        """ code generation """
        self.cu_isa = InstructionSet(500*4, 1)
        self.program_file_name = 'inst-emit.npy'

        """ Patch for the manually compiled part"""
        self.pre_pcg_inst_num = 4
        self.post_pcg_jump_offset = 56

    def insert_ir_table(self, inst_type, 
                     op_type=None, result=None, 
                     v0=None, v1=None, s0=None, s1=None):
        df_insert_row(self.ir_table, [inst_type, op_type, result, v0, v1, s0, s1])

    def add_axpby_frac(self, scalar, vector):
        assert self.axpby_buffer['frac_num'] < 2, \
        "buffer full: {}, {}".format(self.axpby_buffer['v_x'],
                       self.axpby_buffer['v_y'])

        if self.axpby_buffer['frac_num'] == 0:
            self.axpby_buffer['s_a']=scalar
            self.axpby_buffer['v_x']=vector
        elif self.axpby_buffer['frac_num'] == 1:
            self.axpby_buffer['s_b']=scalar
            self.axpby_buffer['v_y']=vector

        self.axpby_buffer['frac_num'] += 1

    def emit_axpby_buffer(self, result_name, op_type='linear'):
        assert self.axpby_buffer['frac_num'] <= 2
        assert self.axpby_buffer['frac_num'] > 0

        self.insert_ir_table(inst_type='axpby', 
                       op_type=op_type,
                       result=result_name,
                       v0=self.axpby_buffer['v_x'],
                       s0=self.axpby_buffer['s_a'],
                       v1=self.axpby_buffer['v_y'],
                       s1=self.axpby_buffer['s_b'])

        """ clear the buffer after emitting """ 
        self.axpby_buffer['s_a']=None
        self.axpby_buffer['v_x']=None
        self.axpby_buffer['s_b']=None
        self.axpby_buffer['v_y']=None
        self.axpby_buffer['frac_num']=0

    def temp_var_info(self, var_type, node):
        if not hasattr(node, 'name'):
            temp_name ='temp-'+str(self.temp_var_idx)
            self.temp_var_idx += 1
            node.name=temp_name
            df_insert_row(self.symbol_table, [node.name, var_type, node.coord, 'unset', None])
        else:
            """ the result of the binop has been declared, 
            check if the declared type and the result type are the same """ 
            decl_type = lookup_var_type(self.symbol_table, node)
            assert decl_type == var_type

    def const_var_info(self, var_type, node):
        """ check if the constant exists """
        df = self.symbol_table
        float_value = float(node.value)
        id_record = df.loc[df['const_value'] == float_value]
        # assert len(id_record) <= 1, print(len(id_record), id_record)
        # if len(id_record) == 1:
        if len(id_record) >= 1:
            """ Note: the src loc of the constant with the same value
                is not recorded """
            node.name = df.at[id_record.index[0], 'id']
        else:
            const_name ='const-'+str(self.const_var_idx)
            self.const_var_idx += 1
            node.name=const_name
            df_insert_row(self.symbol_table, [node.name, var_type, node.coord, 'const', float_value])

    def const_fill_info(self, fill_value):
        """ check if the constant exists """
        df = self.symbol_table
        id_record = df.loc[df['const_value'] == fill_value]
        # assert len(id_record) <= 1, print(len(id_record))
        # if len(id_record) == 1:
        if len(id_record) >= 1:
            """ Note: the src loc of the constant with the same value
                is not recorded """
            const_name = df.at[id_record.index[0], 'id']
        else:
            const_name ='const-'+str(self.const_var_idx)
            self.const_var_idx += 1
            df_insert_row(self.symbol_table, [const_name, 'float', 'filled', 'const', fill_value])

        return const_name
            
    def visit_Decl(self, node):
        """ gather ID, type info from decl stmt"""
        # print(node)
        df_insert_row(self.symbol_table, get_Decl_info(node))

    def visit_Constant(self, node):
        self.const_var_info('float', node)

    def visit_BinaryOp(self, node):
        for c in node:
            self.visit(c)
        """ choose machine idiom based on l-type and r-type """
        left_type = lookup_var_type(self.symbol_table, node.left)
        right_type = lookup_var_type(self.symbol_table, node.right)
        prod_head = (node.op, left_type, right_type)
        emit_inst, op_type = self.binop_prod_rule.get(prod_head)
        assert emit_inst is not None 

        result_type = self.inst_out_type.get(emit_inst)
        self.temp_var_info(result_type, node)

        if emit_inst == 'axpby_frac':
            """ fraction buffer filled flag """
            self.add_axpby_frac(node.left.name, node.right.name)
            node.axpby_frac_flag = True
        elif emit_inst == 'axpby':
            if not hasattr(node.left, 'axpby_frac_flag'):	
                fill_var_name = self.const_fill_info(1.0)
                self.add_axpby_frac(fill_var_name, node.left.name)

            if not hasattr(node.right, 'axpby_frac_flag'):	
                if node.op == '-':
                    fill_scalar = -1.0
                else:
                    fill_scalar = 1.0
                fill_var_name = self.const_fill_info(fill_scalar)
                self.add_axpby_frac(fill_var_name, node.right.name)

            self.emit_axpby_buffer(node.name, op_type)
        elif emit_inst == 'scalar_op':
            self.insert_ir_table(inst_type=emit_inst, 
                            op_type=op_type,
                            result=node.name,
                            s0=node.left.name,
                            s1=node.right.name,)
        elif emit_inst == 'branch':
            if op_type == 'gt':
                self.insert_ir_table(inst_type=emit_inst, 
                                op_type='block_start',
                                s0=node.left.name,
                                s1=node.right.name,)
            elif op_type == 'lt':
                self.insert_ir_table(inst_type=emit_inst, 
                                op_type='block_start',
                                s0=node.right.name,
                                s1=node.left.name,)
            elif op_type == 'OR':
                """ Only support multiple OR conditions now """
                pass
            elif op_type == 'AND':
                assert False, print('''Add support for AND later, \
                    use nested if instead''')

        else:
            self.insert_ir_table(inst_type=emit_inst, 
                            op_type=op_type,
                            result=node.name,
                            v0=node.left.name,
                            v1=node.right.name,)

    def visit_Assignment(self, node):
        """ using the assigment value instead of temp var as name""" 
        assert isinstance(node.rvalue, c_ast.BinaryOp) or isinstance(node.rvalue, c_ast.ID)

        if isinstance(node.rvalue, c_ast.BinaryOp):
            node.rvalue.name = node.lvalue.name
            for c in node:
                self.visit(c)

            if self.axpby_buffer['frac_num'] > 0:
                self.emit_axpby_buffer(node.lvalue.name)

        if isinstance(node.rvalue, c_ast.ID):
            left_type = lookup_var_type(self.symbol_table, node.lvalue)
            right_type = lookup_var_type(self.symbol_table, node.rvalue)
            assert left_type == 'vectorf' and right_type == 'vectorf'
            fill_var_name = self.const_fill_info(1.0)
            self.add_axpby_frac(fill_var_name, node.rvalue.name)
            self.emit_axpby_buffer(node.lvalue.name)

    def visit_FuncCall(self,node):
        func_name = node.name.name
        arg_list = node.args.exprs
        id_list = list(map(lambda x: x.name, arg_list))

        if func_name == 'calc_norm_inf':
            assert len(id_list) ==  2
            self.insert_ir_table(inst_type=func_name, 
                            result=id_list[1],
                            v0=id_list[0])

        if func_name == 'select_max':
            assert len(id_list) ==  3
            self.insert_ir_table(inst_type='scalar_op', 
                            op_type=func_name,
                            s0=id_list[1],
                            s1=id_list[2],
                            result=id_list[0])

        if func_name == 'select_min':
            assert len(id_list) ==  3
            self.insert_ir_table(inst_type='scalar_op', 
                            op_type=func_name,
                            s0=id_list[1],
                            s1=id_list[2],
                            result=id_list[0])

        if func_name == 'c_sqrt':
            assert len(id_list) == 2
            self.insert_ir_table(inst_type='scalar_op', 
                                 op_type=func_name,
                                 s0=id_list[1],
                                 s1=id_list[1],
                                 result=id_list[0])
    def visit_If(self, node):
        """ Add support for the false branch later """
        assert node.iffalse is None

        self.visit(node.cond)
        """ Uncondition jump to block end """
        self.insert_ir_table(inst_type='branch', 
                        op_type='block_end',
                        s0='const_plus_1',
                        s1='const_minus_1',)
        iftrue_start_addr = self.current_inst_addr()
        self.visit(node.iftrue)
        iftrue_end_addr = self.current_inst_addr()

        df = self.ir_table
        for idx in reversed(range(len(df))):
            if df.at[idx, 'inst_type'] == 'branch':
                if df.at[idx, 'result'] is not None:
                    break
                elif df.at[idx, 'op_type'] == 'block_start': 
                    df.at[idx, 'result'] = iftrue_start_addr
                elif df.at[idx, 'op_type'] == 'block_end': 
                    df.at[idx, 'result'] = iftrue_end_addr
                else:
                    assert False, print(df.at[idx, 'op_type'])

    def visit_Compound(self, node):
        for stmts in node.block_items:
            self.visit(stmts)

    def current_inst_addr(self):
        """ Compute the current instruction address of the IR table,
            Each SpMV in the IR table will emit 3 instructions
        """
 
        df = self.ir_table
        id_record = df.loc[df['inst_type'] == 'spmv' ]
        spmv_num = len(id_record)
        current_addr = len(self.ir_table)\
            + self.post_pcg_jump_offset +\
            + spmv_num*2 \
            - self.pre_pcg_inst_num
        return current_addr

    def init_pass(self):
        """ pass 1: scan for variables that need init """
        for _, row in self.ir_table.iterrows():

            """ build a list for the bi_partition pass to use """
            if row['v0'] is not None and row['v1'] is not None and\
                   row['inst_type'] == 'axpby':
                """ add 2 nodes to the bp graph"""
                for item in ['v0', 'v1']:
                    if row[item] not in self.bp_graph_nodes:
                        self.bp_graph_nodes.append(row[item])

                """ add an edge to the bp graph"""
                if (row['v0'], row['v1']) not in self.bp_graph_edges and\
                    (row['v1'], row['v0']) not in self.bp_graph_edges:
                    self.bp_graph_edges.append((row['v0'], row['v1']))
            
            """ build register layout """
            for item in ['s0', 's1']:
                if row[item] not in self.reg_onchip and row[item] is not None:
                    self.reg_onchip.append(row[item])

        for _, row in self.ir_table.iterrows():
            if row['inst_type'] == 'axpby': 
                """ if one operand is None and not used in other dual operand axpby instruction, 
                then it can choose LHS or RHS freely """
                for item in ['v0', 'v1', 'result']:
                    if row[item] not in self.bp_graph_nodes and\
                        row[item] is not None and\
                            row[item] not in self.lhs_or_rhs_hbm:
                        self.lhs_or_rhs_hbm.append(row[item])

            if row['inst_type'] == 'calc_norm_inf' or row['inst_type'] == 'scalar_op': 
                item = 'result'
                assert row[item] is not None
                if row[item] not in self.reg_onchip:
                    self.reg_onchip.append(row[item])

    def bi_partition_pass(self):
        """ pass 2: partition the vectors onto LHS or RHS and assign address
        like the 2-way partition problem mentioned in the Stephen's CVX lecture, book page 220
        minimize x^T*W*x, sub (x_i)^2=1 """

        graph_size = len(self.bp_graph_nodes)	
        """ the W matrix """
        partition_costs = np.zeros((graph_size, graph_size), dtype=np.int32)
        for node_0, node_1 in self.bp_graph_edges:
            index_0 = self.bp_graph_nodes.index(node_0)
            index_1 = self.bp_graph_nodes.index(node_1)
            partition_costs[index_0, index_1] = 1
            partition_costs[index_1, index_0] = 1

        """ solution of the 2 way partition problem, 
        take value -1 or 1, use brute force """
        sol_codec = {'0':-1, '1':1}
        brute_force_cost = np.inf
        solution_space_size = 2**graph_size
        # for sol_item in tqdm(range(solution_space_size)):
        for sol_item in range(solution_space_size):
            """ encode solution in the search space """
            sol_str = format(sol_item, 'b')
            sol_str =(graph_size-len(sol_str)) *'0' + sol_str
            sol_str2list = list(sol_str)
            sol_x = np.array(list(map(lambda x: sol_codec.get(x), sol_str2list)), dtype=np.int32)
            sol_cost = sol_x.transpose() @ partition_costs @ sol_x
            if sol_cost <= brute_force_cost:
                brute_force_sol = sol_x
                brute_force_cost = sol_cost

        """ Check if the solution is a full bipartition graph """
        assert brute_force_cost == len(self.bp_graph_edges)*-2

        """ Generate LHS and RHS memory layout CSV file based on brute_force_sol 
            and pass it to the binary linker """
        for vec_id, vec_loc in zip(self.bp_graph_nodes, brute_force_sol):
            if vec_loc == -1:
                self.rhs_hbm.append(vec_id)
            else:
                self.lhs_hbm.append(vec_id)

        lhs_size = len(self.lhs_hbm)
        rhs_size = len(self.rhs_hbm)
        if lhs_size >= rhs_size:
            self.rhs_hbm += self.lhs_or_rhs_hbm
        else:
            self.lhs_hbm += self.lhs_or_rhs_hbm
        lhs_size = len(self.lhs_hbm)
        rhs_size = len(self.rhs_hbm)

        layout_size = max(lhs_size, rhs_size)
        self.lhs_hbm += (layout_size-lhs_size)* [None]
        self.rhs_hbm += (layout_size-rhs_size)* [None]

        for left_item, right_item in zip(self.lhs_hbm, self.rhs_hbm):
            df_insert_row(self.lr_layout, [left_item, right_item])

        self.lr_layout.to_csv('lr_layout.csv', index=False)
        for idx, row in self.ir_table.iterrows():
            df = self.ir_table
            if row['inst_type'] == 'axpby':
                item_s0, item_s1 = row['s0'], row['s1']
                item_v0, item_v1  = row['v0'], row['v1']

                """ swap the vector in the instruction if the layout says so""" 
                if item_v1 in self.lhs_hbm and item_v1 is not None:
                    assert item_v0 in self.rhs_hbm or item_v0 is None
                    df.at[idx, 'v0'] = item_v1
                    df.at[idx, 'v1'] = item_v0
                    df.at[idx, 's0'] = item_s1
                    df.at[idx, 's1'] = item_s0

                if item_v0 in self.rhs_hbm and item_v0 is not None:
                    assert item_v1 in self.lhs_hbm or item_v1 is None
                    df.at[idx, 'v0'] = item_v1
                    df.at[idx, 'v1'] = item_v0
                    df.at[idx, 's0'] = item_s1
                    df.at[idx, 's1'] = item_s0

    def lookup_lr_addr(self, row_item):
        assert row_item in self.lhs_hbm or row_item in self.rhs_hbm or row_item is None
        if row_item is None:
            sel_lhs = 0
            lr_addr = 0
        elif row_item in self.lhs_hbm:
            sel_lhs = 1
            lr_addr = self.lhs_hbm.index(row_item) + self.vec_addr_offset
        elif row_item in self.rhs_hbm:
            sel_lhs = 0
            lr_addr = self.rhs_hbm.index(row_item) + self.vec_addr_offset

        return lr_addr, sel_lhs

    def lookup_reg_addr(self, row_item):
        if row_item in self.reg_onchip:
            return self.reg_onchip.index(row_item)
        else:
            return 0

    def lookup_id_value(self,  id):
        df = self.symbol_table
        id_record = df.loc[df['id'] == id]
        """ the ID should be unique """
        assert len(id_record) <= 1, print(id)
        if len(id_record) == 1:
            id_value = id_record.iloc[0]['const_value']
        else:
            id_value = 0.0
        return id_value

    def codegen_pass(self):
        """ Generate register init value """
        rf_file_size=64
        cu_register_file = np.zeros(rf_file_size, dtype=np.float32)
        for idx, item in enumerate(self.reg_onchip):
            item_value = self.lookup_id_value(item)
            df_insert_row(self.reg_layout, [item, item_value])
            if item_value is not None:
                cu_register_file[idx]=item_value

        self.reg_layout.to_csv('reg_layout.csv', index=False)
        np.save('register_file.npy',cu_register_file)

        """ Pass the address of work_xtilde_view and work_ztild_view to PCG code """
        self.cu_dict['work_xtilde_view_addr'] = self.lookup_lr_addr('work_xtilde_view')
        self.cu_dict['work_ztilde_view_addr'] = self.lookup_lr_addr('work_ztilde_view')
        self.cu_dict['pcg_rhs_part1_addr'] = self.lookup_lr_addr('pcg_rhs_part1')
        self.cu_dict['pre_pcg_inst_num'] = self.pre_pcg_inst_num
        self.cu_dict['post_pcg_jump_offset']=self.post_pcg_jump_offset

        for _, row in self.ir_table.iterrows():
            if row['inst_type'] == 'axpby':

                assert row['v0'] in self.lhs_hbm or row['v0'] is None
                assert row['v1'] in self.rhs_hbm or row['v1'] is None

                alpha_addr = self.lookup_reg_addr(row['s0'])
                beta_addr = self.lookup_reg_addr(row['s1'])

                src_lhs_addr, _ = self.lookup_lr_addr(row['v0'])
                src_rhs_addr, _ = self.lookup_lr_addr(row['v1'])

                assert row['result'] in self.lhs_hbm or row['result'] in self.rhs_hbm 
                dst_addr, dst_sel_lhs = self.lookup_lr_addr(row['result'])

                assert row['op_type'] in self.axpby_type_dict
                op_type = self.axpby_type_dict.get(row['op_type'])

                self.cu_isa.add_axpby_inst(alpha_addr=alpha_addr,
                                    beta_addr=beta_addr,
                                    src_tetris_addr=src_lhs_addr,
                                    src_vf_addr=src_rhs_addr,
                                    dst_sel_tetris=dst_sel_lhs,
                                    dst_addr=dst_addr,
                                    pack_size=self.cu_dict['unified_vec_pack_len'],
                                    op_type=op_type)

            if row['inst_type'] == 'spmv':
                assert row['v0'] in self.mat_id_dict
                mat_id, pack_len_key = self.mat_id_dict.get(row['v0'])
                mat_name = row['v0'].split('_')[1]

                """ Load tetris"""
                assert row['v1'] in self.lhs_hbm or row['v1'] in self.rhs_hbm
                src_addr, src_sel_lhs = self.lookup_lr_addr(row['v1'])
                location_map_offset = self.cu_dict['offset_'+mat_name+'_location_map']
                pack_size=self.cu_dict[pack_len_key]
                self.cu_isa.add_load_tetris_inst(src_addr=src_addr,
                                                 src_sel_lhs=src_sel_lhs,
                                                 location_map_offset=location_map_offset,
                                                 pack_size=pack_size)

                """ Duplicate vector """
                tetris_height=self.cu_dict['tetris_height_'+mat_name]
                duplicate_map_offset = self.cu_dict['offset_'+mat_name+'_duplicate_map']
                self.cu_isa.add_complete_tetris_inst(tetris_height=tetris_height,
                                                     duplicate_map_offset=duplicate_map_offset)

                """ Do SpMV """
                assert row['result'] in self.lhs_hbm or row['result'] in self.rhs_hbm
                dst_addr, dst_sel_lhs = self.lookup_lr_addr(row['result'])
                self.cu_isa.add_spmv_inst(mat_id=mat_id, 
                                          dst_addr=dst_addr,
                                          dst_sel_lhs=dst_sel_lhs)

            if row['inst_type'] == 'calc_norm_inf':
                assert row['v0'] in self.lhs_hbm or row['v0'] in self.rhs_hbm
                src_addr, src_sel_lhs = self.lookup_lr_addr(row['v0'])
                assert row['result'] in self.reg_onchip
                dst_addr = self.lookup_reg_addr(row['result'])
                self.cu_isa.add_norm_inf_inst(src_addr, 
                                              dst_addr,
                                              src_sel_lhs,
                                              self.cu_dict['unified_vec_pack_len'])

            if row['inst_type'] == 'scalar_op':
                assert row['s0'] in self.reg_onchip and\
                    row['s1'] in self.reg_onchip and\
                    row['result'] in self.reg_onchip
                src_0_reg = self.lookup_reg_addr(row['s0'])
                src_1_reg = self.lookup_reg_addr(row['s1'])
                dst_reg = self.lookup_reg_addr(row['result'])
                imme_flag=0

                assert row['op_type'] in self.scalar_op_type_dict
                op_type = self.scalar_op_type_dict.get(row['op_type'])
                self.cu_isa.add_scalar_op_inst(src_0_reg, 
                                               src_1_reg,
                                               op_type,
                                               dst_reg,
                                               imme_flag)

            if row['inst_type'] == 'branch':
                assert row['s0'] in self.reg_onchip and\
                    row['s1'] in self.reg_onchip 
                src_0_reg = self.lookup_reg_addr(row['s0'])
                src_1_reg = self.lookup_reg_addr(row['s1'])
                jump_address= row['result']
                self.cu_isa.add_branch_inst(src_0_reg,
                                            src_1_reg,
                                            jump_address,
                                            imme_flag=0,
                                            imme_number=0)

        """ Pass the [op0, op1, op2, op3] array to link_binary.py """
        np.save(self.program_file_name,
                 self.cu_isa.uint32_data_region[0:self.cu_isa.uint32_data_pointer])

def compile_update_xzy(cu_dict):
    filename='./osqp_alg_desc.c'
    ast = parse_file(filename, use_cpp=False)
    # ast.show(showcoord=False)
    main_stmts = ast.ext[0].body.children()

    ev = EmitVisitor(cu_dict)
    for _, stmt in enumerate(main_stmts):
        ev.visit(stmt[1])

    ev.init_pass()
    ev.bi_partition_pass()

    # print(ev.ir_table)
    ev.ir_table.to_csv('ir_table.csv', index=False)
    # print(ev.lr_layout)

    ev.codegen_pass()

    # print(ev.symbol_table)
    ev.symbol_table.to_csv('symbol_table.csv', index=False)
    # df = ev.symbol_table
    # print(df.loc[(df['init_flag']=='yes') & (df['type']=='vectorf')])
    # print(df.loc[(df['type']=='float')])
    # print(df[df['init_flag']=='no'])
    # print(df[df['init_flag']=='unset'])
    # print(df[df['init_flag']=='const'])
    
if __name__ == "__main__":
    compile_update_xzy()