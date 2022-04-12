"""
Main tester script
"""
"""
Imports
"""
import os
import numpy as np
import pickle
import time
from data_constructor import Dataset
from ioi_constructor import IOI
from global_model_constructor import Global_model
from perturbator_constructor_bosse import Perturbator
from evaluator_constructor import Evaluator
path_here = os.path.abspath('')
results_cf_obj_dir = str(path_here)+'/Results/local_obj/'
results_grid_search = str(path_here)+'/Results/grid_search/'

def save_obj(evaluator_obj,file_name):
    """
    Method to store an Evaluator object containing the evaluation results for all the instances of a given dataset
    """
    with open(results_cf_obj_dir+file_name, 'wb') as output:
        pickle.dump(evaluator_obj, output, pickle.HIGHEST_PROTOCOL)

seed_int = 54321
k = 50
N = 300
regul = 0.1
x_iterations = 3
global_model_type = 'linear'                               # Possible values = ['linear','nonlinear']. Nonlinear model must have predict_proba() function
datasets = ['synthetic_circle','3-Cubic','sinusoid']     # ['synthetic_circle','3-Cubic','sinusoid','synthetic_diagonal_0','synthetic_diagonal_1_8','synthetic_diagonal_plane','compass','ionosphere','german']
start_all_time = time.time()
for data_str in datasets:
    data = Dataset(data_str,0.7,seed_int)
    eval = Evaluator(data, x_iterations)
    global_model = Global_model(global_model_type,data)
    data_test_pd_range = range(5)
    for idx in data_test_pd_range:
        x = data.test_pd.iloc[idx,:]
        x_normal = data.processed_test_pd.iloc[idx,:]
        ioi = IOI(idx, x, x_normal, data, k)
        perturbator_type_list = ['bosse']
        for i in perturbator_type_list:
            perturbator_type = i
            perturbator_i = Perturbator(i, N, ioi, data, global_model, global_model_type, seed_int, regul)
            eval.add_perturbator_eval(perturbator_i)
            print(f'Dataset: {data_str}, Index: {idx}, Perturbator: {i}, Time (s): {np.round_(perturbator_i.exec_time,2)}')
        save_obj(eval, data_str+'_eval.pkl')
end_all_time = time.time()
print(f'Total time (s): {np.round_(end_all_time - start_all_time,2)}')