import os
import sys
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
os.system("python source/spilt1.py")
sys.path.append('source')
import prob1

data = pd.read_csv('data/training.csv')
data_t = pd.read_csv('data/test.csv')
data_x = data.drop(['ID', 'G1', 'G2', 'G3', 'cat_mat', 'cat_por'], axis=1)
data_x_t = data_t.drop(['ID', 'G1', 'G2', 'G3', 'cat_mat', 'cat_por'], axis=1)
data_y = data['G3']
data_y_t = data_t['G3']

X = data_x.values
X_t = data_x_t.values
Y = data_y.values
Y_t = data_y_t.values
Y = Y.reshape((800, 1))
Y_t = Y_t.reshape((200, 1))

RMSE1, p1 = prob1.lr_nb_ps(X, Y, X_t, Y_t)
RMSE2, p2 = prob1.lr_nb(X, Y, X_t, Y_t)
RMSE3, p3 = prob1.lr(X, Y, X_t, Y_t)
RMSE4, p4 = prob1.lr_bay(X, Y, X_t, Y_t)

plt.plot(Y_t, color='blue', label='ground truth')
plt.plot(p1, color='orange', label='(%.2f) Linear Regression' %RMSE1)
plt.plot(p2, color='green', label='(%.2f) Linear Regression (reg)' %RMSE2)
plt.plot(p3, color='red', label='(%.2f)Linear Regression (r/b)' %RMSE3)
plt.plot(p4, color='purple', label='(%.2f) Bayseian Linear Regression' %RMSE4)
plt.legend()
plt.title('Figure 1: Regression result comparison.')
plt.ylabel('Values')
plt.xlabel('Sample Index')
plt.show()


os.system("python source/spilt2.py")
import prob2

data = pd.read_csv('data/training.csv')
data_t = pd.read_csv('data/test.csv')
data_x = data.drop(['ID', 'G1', 'G2', 'G3', 'cat_mat', 'cat_por', 'label'], axis=1)
data_x_t = data_t.drop(['ID', 'G1', 'G2', 'G3', 'cat_mat', 'cat_por', 'label'], axis=1)
data_y = data['label']
data_y_t = data_t['label']

X = data_x.values
X_t = data_x_t.values
Y = data_y.values
Y_t = data_y_t.values
Y = Y.reshape((800, 1))
Y_t = Y_t.reshape((200, 1))

p1_1 = prob2.lr(X, Y, X_t, Y_t, 0.1)
p1_2 = prob2.lr(X, Y, X_t, Y_t, 0.5)
p1_3 = prob2.lr(X, Y, X_t, Y_t, 0.9)

p2_1 = prob2.lgr(X, Y, X_t, Y_t, 0.1)
p2_2 = prob2.lgr(X, Y, X_t, Y_t, 0.5)
p2_3 = prob2.lgr(X, Y, X_t, Y_t, 0.9)

con = prob2.con_mat(p1_2, Y_t)
sns.heatmap(con, linewidths=.5, annot=True, cbar=False, yticklabels=['predict = 0', 'predict = 1'])
plt.show()

con = prob2.con_mat(p2_2, Y_t)
sns.heatmap(con, linewidths=.5, annot=True, cbar=False, yticklabels=['predict = 0', 'predict = 1'])
plt.show()

con = prob2.con_mat(p1_3, Y_t)
sns.heatmap(con, linewidths=.5, annot=True, cbar=False, yticklabels=['predict = 0', 'predict = 1'])
plt.show()

con = prob2.con_mat(p2_3, Y_t)
sns.heatmap(con, linewidths=.5, annot=True, cbar=False, yticklabels=['predict = 0', 'predict = 1'])
plt.show()

import prob3

data = pd.read_csv('data/training.csv')
data_t = pd.read_csv('data/test_no_G3.csv')
data_x = data.drop(['ID', 'G1', 'G2', 'G3', 'cat_mat', 'cat_por', 'label'], axis=1)
data_x_t = data_t.drop(['ID', 'G1', 'G2', 'cat_mat', 'cat_por'], axis=1)
data_y = data['G3']

X = data_x.values
X_t = data_x_t.values
Y = data_y.values
Y = Y.reshape((800, 1))

prob3.p1(X, Y, X_t)
				
data = pd.read_csv('data/training.csv')
data_t = pd.read_csv('data/test_no_G3.csv')
data_x = data.drop(['ID', 'G1', 'G2', 'G3', 'cat_mat', 'cat_por', 'label'], axis=1)
data_x_t = data_t.drop(['ID', 'G1', 'G2', 'cat_mat', 'cat_por'], axis=1)
data_y = data['label']

X = data_x.values
X_t = data_x_t.values
Y = data_y.values
Y = Y.reshape((800, 1))

prob3.p2(X, Y, X_t)											