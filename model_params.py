"""
Imports
"""

import ast
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import copy

def best_model_params(grid_search_pd,data_str):
    """
    Method that delivers the best model and its parameters according to the Grid Search done
    Input grid_search_pd: DataFrame containing the parameters of the models tested in the Grid Search
    Input data_str: String containing the name of the dataset
    Output best: The name of the best performing model
    Output params_best: The parameters of the best performing model
    Output params_rf: The parameters of the RF model
    """
    if data_str in ['sinusoid','ionosphere','german','credit']:
        best = 'rf'
    elif data_str in ['synthetic_circle','piecewise','synthetic_diagonal_0',
                      'synthetic_diagonal_1_8','synthetic_diagonal_plane','synthetic_cubic_0','synthetic_cubic_1_8','compass']:
        best = 'mlp'
    params_best = ast.literal_eval(grid_search_pd.loc[(data_str,best),'params'])[0]
    return best, params_best

def clf_model(model_str,best_params,train_data,train_target):
    """
    Method that outputs the best trained model according to Grid Search done
    Input model_str: The name of the best performing model
    Input best_params: Parameters of the best performing model
    Input rf_params: Parameters of the RF model
    Input train_data: Training dataset
    Input train_target: Target of the training dataset
    Output model: Trained best performing model
    """
    random_st = 54321 
    if model_str == 'mlp':
        best_activation = best_params['activation']
        best_hidden_layer_sizes = best_params['hidden_layer_sizes']
        best_solver = best_params['solver']
        best_model = MLPClassifier(activation=best_activation, hidden_layer_sizes=best_hidden_layer_sizes, solver=best_solver, random_state=random_st)
        best_model.fit(train_data,train_target)
    elif model_str == 'rf':
        best_max_depth = best_params['max_depth']
        best_min_samples_leaf = best_params['min_samples_leaf']
        best_min_samples_split = best_params['min_samples_split']
        best_n_estimators = best_params['n_estimators']
        best_model = RandomForestClassifier(max_depth=best_max_depth, min_samples_leaf=best_min_samples_leaf, min_samples_split=best_min_samples_split, n_estimators=best_n_estimators)
        best_model.fit(train_data,train_target) 
    return best_model