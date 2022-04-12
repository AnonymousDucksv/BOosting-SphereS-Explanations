"""
Global model constructor
"""

"""
Imports
"""

import numpy as np
import pandas as pd
import os
from sklearn.linear_model import RidgeClassifier
from model_params import best_model_params, clf_model
path_here = os.path.abspath('')
results_grid_search = str(path_here)+'/Results/grid_search/'

class Global_model:
    """
    Class that holds real and black-box models for the loaded dataset.
    Attributes:
        (1) data:              Data object,
        (2) regul:             Regularization parameter,
        (3) scaler:            Scaler for the continuous features,
        (4) global_model_type: Type of global model to run (Linear or Nonlinear),
        (5) global_model:      Global model trained on the processed training dataset
    """
    def __init__(self,global_model_type,data_obj,regul=0.1) -> None:
        self.data = data_obj
        self.regul = regul
        self.scaler = data_obj.con_scaler
        self.global_model_type = global_model_type
        self.global_model = self.sklearn_global_model_construct(self.regul)    
    
    def sklearn_global_model_construct(self,regul):
        """
        Constructs a global model for the dataset using the sklearn modules
        Input regul: Regularization parameter set up as 0.1 by default
        Output global_model: Global model trained on the processed training dataset
        """
        if self.global_model_type == 'linear':
            global_model = RidgeClassifier(alpha=regul)
            global_model.fit(self.data.processed_train_pd,self.data.train_target)
        else:
            grid_search_results = pd.read_csv(results_grid_search+'grid_search_final_bosse.csv',index_col = ['dataset','model'])
            sel_model_str, params_best = best_model_params(grid_search_results,self.data.name)
            global_model = clf_model(sel_model_str,params_best,self.data.processed_train_pd,self.data.train_target)
        return global_model
    
    def labeler(self,f):
        """
        Function that labels a generated data point
        Input f: ground-truth label function value
        Output f: Binarized ground-truth label function value
        """
        f[f > 0] = 1
        f[f <= 0] = 0
        return f

    def pred_circle(self,x):
        """
        Function that labels synthetic_circle points
        Input x: Instance of interest
        Output label_x: Binarized ground-truth label value
        """
        # x_inverse = self.scaler.inverse_transform(x)
        x_inverse = 8*x - 4
        fx = x_inverse[:,0]**2 + x_inverse[:,1]**2 - 4
        label_x = self.labeler(fx)
        return label_x
    
    def pred_3_Cubic(self,x):
        """
        Function that labels 3-Cubic dataset
        Input x: Instance of interest
        Output label_x: Binarized ground-truth label value
        """
        x_inverse = x*200 - 100
        fx = x_inverse[:,0]**3 - 2*(x_inverse[:,1])**2 + 3*x_inverse[:,2]
        label_x = self.labeler(fx)
        return label_x

    def pred_sinusoid(self,x):
        """
        Function that labels sinusoid dataset
        Input x: Instance of interest
        Output label_x: Binarized ground-truth label value
        """
        # x_inverse = self.scaler.inverse_transform(x)
        x_inverse = 20*x -10 
        fx = x_inverse[:,1] - x_inverse[:,0]*np.sin(x_inverse[:,0])**2
        label_x = self.labeler(fx)
        return label_x

    def pred_synthetic_diagonal_0(self,x):
        """
        Function that labels synthetic_diagonal_0 dataset
        Input x: Instance of interest
        Output label_x: Binarized ground-truth label value
        """
        fx = x[:,1] - x[:,0]
        label_x = self.labeler(fx)
        return label_x

    def pred_synthetic_diagonal_1_8(self,x):
        """
        Function that labels synthetic_diagonal_1_8 dataset
        Input x: Instance of interest
        Output label_x: Binarized ground-truth label value
        """
        fx = x[:,1] - x[:,0] - 1/8
        label_x = self.labeler(fx)
        return label_x

    def pred_synthetic_diagonal_plane(self,x):
        """
        Function that labels synthetic_diagonal_plane dataset
        Input x: Instance of interest
        Output label_x: Binarized ground-truth label value
        """
        fx = 4*x[:,2] + 2*x[:,0] - 3
        label_x = self.labeler(fx)
        return label_x

    def pred_synthetic_cubic_0(self,x):
        """
        Function that labels synthetic_cubic_0 dataset
        Input x: Instance of interest
        Output label_x: Binarized ground-truth label value
        """
        fx = x[:,1] - x[:,0]**3
        label_x = self.labeler(fx)
        return label_x

    def pred_synthetic_cubic_1_8(self,x):
        """
        Function that labels synthetic_cubic_1_8 dataset
        Input x: Instance of interest
        Output label_x: Binarized ground-truth label value
        """
        fx = x[:,1] - x[:,0]**3 - 1/8
        label_x = self.labeler(fx)
        return label_x

    def call_predict(self,x):
        """
        Calls the predict function if it exists on the global model
        """
        if self.data.name == 'synthetic_circle':
            prediction = self.pred_circle(x)
        elif self.data.name == '3-Cubic':
            prediction = self.pred_3_cubic(x)
        elif self.data.name == 'sinusoid':
            prediction = self.pred_sinusoid(x)
        elif self.data.name == 'synthetic_diagonal_0':
            prediction = self.pred_synthetic_diagonal_0(x)
        elif self.data.name == 'synthetic_diagonal_1_8':
            prediction = self.pred_synthetic_diagonal_1_8(x)
        elif self.data.name == 'synthetic_diagonal_plane':
            prediction = self.pred_synthetic_diagonal_plane(x)
        elif self.data.name == 'synthetic_cubic_0':
            prediction = self.pred_synthetic_cubic_0(x)
        elif self.data.name == 'synthetic_cubic_1_8':
            prediction = self.pred_synthetic_cubic_1_8(x)
        else:
            prediction = self.global_model.predict(x)
        return prediction
        
    def call_predict_proba(self,x):
        """
        Calls the predict_proba function for the nonlinear model (it is only called when the model has the predict_proba function)
        Input x: Instance of interest
        Output prediction_proba: Vector with each class's prediction probability
        """
        prediction_proba = self.global_model.predict_proba(x)
        return prediction_proba
    
    def call_decision_function(self,x):
        """
        Calls the decision_function function for the linear model (it is only called when the model has the decision_function function)
        Input x: Instance of Interest
        Output decision_function: The decision function carrying the distance from the instance of interest to the decision boundary
        """
        decision_function = self.global_model.decision_function(x)
        return decision_function
    
    def call_coefficients(self,ioi_obj,processed_train,train_target):
        """
        Calculates the coefficients of a linear approximation (if the global model is complex) around an instance of interest using only kNN observations with ground truth label
        Input ioi_obj: Instance of Interest object
        Input processed_train: Processed training dataset
        Input train_target: Training dataset targets
        Output coefficients: Coefficients of the global model (either linear or nonlinear, if the latter, it uses a nearest training neighborhood linear model)
        """
        if self.global_model_type == 'linear':
            coefficients = self.global_model.coef_
        else:
            k = ioi_obj.k 
            normal_x_kNN = ioi_obj.x_kNN
            unique_kNN_labels = np.unique([i[1] for i in normal_x_kNN])
            if len(unique_kNN_labels) == 1:
                distance = []
                for i in range(processed_train.shape[0]):
                    if train_target.iloc[i] != unique_kNN_labels:
                        dist = np.sqrt(np.sum((processed_train.iloc[i,:]-ioi_obj.normal_x)**2))
                        distance.append((processed_train.iloc[i,:],train_target.iloc[i],dist))
                    if len(distance) == 10:
                        break
                distance.sort(key=lambda x: x[-1])
                normal_x_kNN.extend(distance)
            train_kNN_instances = np.array([i[0] for i in normal_x_kNN])
            train_kNN_targets = np.array([i[1] for i in normal_x_kNN])
            lin_model = RidgeClassifier(alpha=self.regul)
            lin_model.fit(train_kNN_instances,train_kNN_targets)
            coefficients = lin_model.coef_
        return coefficients

                