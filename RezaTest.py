
import numpy as np
from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def matrixTranspose(A):
    Atrans = np.transpose(A)
    return Atrans

def matrixInv(A):
    AInv = np.inverse(A)
    return AInv

def matrixProduct(A,B):
    AProduct = np.dot(A,B)
    return AProduct

def optimalTeta (X,Y):
    return matrixProduct(matrixInv(matrixProduct(matrixTranspose(X), X)),(matrixProduct(matrixTranspose(X), Y)))



