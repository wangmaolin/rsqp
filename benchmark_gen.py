import sys
import numpy as np

sys.path.append('/vol/datastore/maolin/osqp_benchmarks')
sys.path.append('/demo/osqp_benchmarks')

from problem_classes.random_qp import RandomQPExample
from problem_classes.eq_qp import EqQPExample
from problem_classes.portfolio import PortfolioExample
from problem_classes.lasso import LassoExample
from problem_classes.svm import SVMExample
from problem_classes.huber import HuberExample
from problem_classes.control import ControlExample
# from app_pat_gen import gen_int_log_space

def gen_int_log_space(min_val, limit, n):
    result = [1]
    if n > 1:  # just a check to avoid ZeroDivisionError
        ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
    while len(result) < n:
        next_value = result[-1]*ratio
        if next_value - result[-1] >= 1:
            # safe zone. next_value will be a different integer
            result.append(next_value)
        else:
            # problem! same integer. we need to find next_value
            # by artificially incrementing previous value
            result.append(result[-1]+1)
            # recalculate the ratio so that the remaining values will scale
            # correctly
            ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
    # round, re-adjust to 0 indexing (i.e. minus 1) and return np.uint64 array
    return np.array(list(map(lambda x: round(x)-1 + min_val, result)),
                    dtype=int)

n_dim = 20
problem_dimensions = {'Random QP': gen_int_log_space(10, 2000, n_dim),
					  'Eq QP': gen_int_log_space(10, 2000, n_dim),
					  'Portfolio': gen_int_log_space(5, 150, n_dim),
					  'Lasso': gen_int_log_space(10, 200, n_dim),
					  'SVM': gen_int_log_space(10, 200, n_dim),
					  'Huber': gen_int_log_space(10, 200, n_dim),
					  'Control': gen_int_log_space(10, 100, n_dim)}

examples = [RandomQPExample,
			EqQPExample,
			PortfolioExample,
			LassoExample,
			SVMExample,
			HuberExample,
			ControlExample]

EXAMPLES_MAP = {example.name(): example for example in examples}

def problem_instance_gen(test_problem_name='Control', dim_idx = 0, random_seed = 0):
	np.random.seed(random_seed)
	dimension = problem_dimensions[test_problem_name][dim_idx]
	example_instance = EXAMPLES_MAP[test_problem_name](dimension, 1)
	return example_instance.qp_problem