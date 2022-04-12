"""
True explanation extractor for datasets
Based partially on:
Jia, Y., Bailey, J., Ramamohanarao, K., Leckie, C., Houle, M.E.: Improving the Quality of Explanations with Local Embedding Perturbations, Knowledge Discovery in Databases (2019)
"""

"""
Imports
"""

import numpy as np
import pandas as pd
import math
from scipy.optimize import minimize

def expf_2(x):
    """
    Function used for true explanation extraction from synthetic dataset 3.
    Input x: Array of size 3 with numerical features between -100 and 100 inclusive, corresponding to the instance at the function f2 = 0 which is closest to the instance of interest.
    Output expf2_array: explanation weight vector as array.
    """
    expf2 = [3*x[0][0]**2,-4*x[0][1],3]
    expf2_array = np.array(expf2)
    return expf2_array

def expf_3(x):
    """
    Function used for true explanation extraction from synthetic dataset 4.
    Input x: Array of size 2 with numerical features between -10 and 10 inclusive, corresponding to the instance at the function f3 = 0 which is closest to the instance of interest.
    Output expf3_array: explanation weight vector as array.
    """
    expf3 = [-math.sin(x[0][0])*(math.sin(x[0][0])+2*x[0][0]*math.cos(x[0][0])),1]
    expf3_array = np.array(expf3)
    return expf3_array

def expf_circle(x):
    """
    Function used for true explanation extraction from synthetic dataset circle.
    Input x: Array of size 2 with numerical features between -4 and 4 inclusive, corresponding to the instance at the function circle = 0 which is closest to the instance of interest.
    Output expfcircle_array: explanation weight vector as array.
    """
    expfcircle = [2*x[0][0],2*x[0][1]]
    expfcircle_array = np.array(expfcircle)
    return expfcircle_array

def expf_diagonal():
    return np.array([-1,1])

def expf_diagonal_plane():
    return np.array([2,0,4])

def expf_cubic_0(x):
    return np.array([-3*x[0][0]**2,1])

def expf_cubic_1_8(x):
    return np.array([-3*x[0][0]**2,1])

def distance_to_f2(x,point3D):
    """
    Function used for distance minimization from point3D to function f2.
    Input x: Array of closest instance in f2 to instance of interest for explanation to be optimized.
    Input point3D: 3-Dimensional point representing the instance of interest for the f2 function.
    Output closest_point: coordinates of the closest point for function 2 to the instance of interest.
    """
    return (x[0] - point3D[0])**2 + (x[1] - point3D[1])**2 + ((-x[0]**3)/3+((2/3)*x[1]**2)-point3D[2])**2
        
def minimize_closest_point_f2(target_instance_f2):
    """
    Function used for closest point extraction for function f2.
    target_instance_f2: 3-Dimensional point representing the instance of interest for the f2 function.
    Output closest_point: coordinates of the closest point for function 2 to the instance of interest.
    """
    start_point = [0,0]
    sol = minimize(distance_to_f2,x0 = start_point,args=target_instance_f2)
    closest_x = [sol.x[0],sol.x[1],-(1/3)*sol.x[0]**3 + (2/3)*sol.x[1]**2]
    return closest_x

def distance_to_f3(x,point3D):
    """
    Function used for distance minimization from point3D to function f3.
    Input x: Array of closest instance in f3 to instance of interest for explanation to be optimized.
    Input point3D: 3-Dimensional point representing the instance of interest for the f3 function.
    Output closest_point: coordinates of the closest point for function 3 to the instance of interest.
    """
    return (x[0] - point3D[0])**2 + (x[0]*math.sin(x[0])**2 - point3D[1])**2
        
def minimize_closest_point_f3(target_instance_f3):
    """
    Function used for closest point extraction for function f3.
    target_instance_f3: 3-Dimensional point representing the instance of interest for the f3 function.
    Output closest_point: coordinates of the closest point for function 3 to the instance of interest.
    """
    start_point = target_instance_f3[0]
    sol = minimize(distance_to_f3,x0 = start_point,args=target_instance_f3)
    closest_x = [sol.x[0],sol.x[0]*(math.sin(sol.x[0])**2)]
    return closest_x

def plot_values(data_str):
    """
    Method that returns two vectors: values_x1 and values_x2 for real decision boundary in original feature space. 
    """
    values_x3 = None
    if data_str == 'sinusoid':
        values_x1 = np.linspace(-10,10,100)
        values_x2 = values_x1*(np.sin(values_x1)**2)
    elif data_str == 'synthetic_circle':
        angle = np.linspace( 0 , 2 * np.pi , 100 ) 
        radius = 2
        values_x1 = radius * np.cos(angle) 
        values_x2 = radius * np.sin(angle)
    elif data_str == 'synthetic_diagonal_0':
        values_x1 = np.linspace(0,1)
        values_x2 = values_x1
    elif data_str == 'synthetic_diagonal_1_8':
        values_x1 = np.linspace(0,1)
        values_x2 = values_x1 + 1/8
    elif data_str == 'synthetic_cubic_0':
        values_x1 = np.linspace(0,1)
        values_x2 = 1*values_x1**3
    elif data_str == 'synthetic_cubic_1_8':
        values_x1 = np.linspace(0,1)
        values_x2 = 1*values_x1**3 + 1/8
    elif data_str == 'synthetic_diagonal_plane':
        values_x1 = np.linspace(0,1)
        values_x2 = np.linspace(0,1)
        values_x3 = -0.5*values_x1 + 3/4
    values_x1, values_x2 = values_x1.reshape(-1,1), values_x2.reshape(-1,1)
    if values_x3 is not None:
        values_x1, values_x2, values_x3 = values_x1.reshape(-1,1), values_x2.reshape(-1,1), values_x3.reshape(-1,1) 
        x1_x2 = np.concatenate((values_x1,values_x2,values_x3),axis=1)
    else:
        x1_x2 = np.concatenate((values_x1,values_x2),axis=1)
    return x1_x2

def minimize_closest_point_circle(target_instance_circle):
    """
    Function used for closest point extraction for function circle.
    target_instance_circle: 2-Dimensional point representing the instance of interest for the f3 function.
    Output closest_point: coordinates of the closest point for function 3 to the instance of interest.
    """
    circle_center, radius = np.array([0,0]), 2
    vector = target_instance_circle - circle_center
    vector_magnitude = np.linalg.norm(vector)
    norm_vector = vector / vector_magnitude
    closest_x = circle_center + norm_vector*radius
    return closest_x

def minimize_closest_point_diagonal_0(target_instance_diagonal_0):
    """
    Function used for closest point extraction for function diagonal with intercept zero.
    target_instance_diagonal_0: 2-Dimensional point representing the instance of interest for the diagonal with intercept zero function.
    Output closest_point: coordinates of the closest point for diagonal with intercept zero to the instance of interest.
    """
    x1 = (target_instance_diagonal_0[0]+target_instance_diagonal_0[1])/2
    closest_x = np.array([x1, x1])
    return closest_x

def minimize_closest_point_diagonal_1_8(target_instance_diagonal_1_8):
    """
    Function used for closest point extraction for function diagonal with intercept 1/8.
    target_instance_diagonal_1_8: 2-Dimensional point representing the instance of interest for the diagonal with intercept 1/8 function.
    Output closest_point: coordinates of the closest point for diagonal with intercept 1/8 to the instance of interest.
    """
    x1 = (target_instance_diagonal_1_8[0]+target_instance_diagonal_1_8[1])/2 - 1/16
    x2 = (target_instance_diagonal_1_8[0]+target_instance_diagonal_1_8[1])/2 + 1/16
    closest_x = np.array([x1, x2])
    return closest_x

def minimize_closest_point_diagonal_plane(x):
    if x[0] == 0:
        closest_x = np.array([0,x[1],3/4])
    elif x[0] == 1:
        closest_x = np.array([0,x[1],1/4])
    return closest_x

def distance_to_cubic_0(x,point2D):
    """
    Function used for distance minimization from point2D to cubic function.
    Input x: Array of closest instance in cubic function to instance of interest for explanation to be optimized.
    Input point2D: 2-Dimensional point representing the instance of interest for the f3 function.
    Output closest_point: coordinates of the closest point for cubic function with zero intercept to the instance of interest.
    """
    return (x[0] - point2D[0])**2 + (1*x[0]**3 - point2D[1])**2
        
def minimize_closest_point_cubic_0(target_instance_cubic):
    """
    Function used for closest point extraction for cubic function.
    target_instance_cubic: 2-Dimensional point representing the instance of interest for the cubic function with zero intercept.
    Output closest_point: coordinates of the closest point for cubic function with zero intercept to the instance of interest.
    """
    start_point = target_instance_cubic[0]
    sol = minimize(distance_to_cubic_0,x0 = start_point,args=target_instance_cubic)
    closest_x = [sol.x[0],1*sol.x[0]**3]
    return closest_x

def distance_to_cubic_1_8(x,point2D):
    """
    Function used for distance minimization from point2D to cubic function.
    Input x: Array of closest instance in cubic function to instance of interest for explanation to be optimized.
    Input point2D: 2-Dimensional point representing the instance of interest for the f3 function.
    Output closest_point: coordinates of the closest point for cubic function with zero intercept to the instance of interest.
    """
    return (x[0] - point2D[0])**2 + (1*x[0]**3 + 1/8 - point2D[1])**2
        
def minimize_closest_point_cubic_1_8(target_instance_cubic):
    """
    Function used for closest point extraction for cubic function.
    target_instance_cubic: 2-Dimensional point representing the instance of interest for the cubic function with zero intercept.
    Output closest_point: coordinates of the closest point for cubic function with zero intercept to the instance of interest.
    """
    start_point = target_instance_cubic[0]
    sol = minimize(distance_to_cubic_1_8,x0 = start_point,args=target_instance_cubic)
    closest_x = [sol.x[0],1*sol.x[0]**3 + 1/8]
    return closest_x