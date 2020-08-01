import pandas as pd
import numpy as np

# normal distribution
# x: feature value
# y: class

def disfun(x, prior, m, v):
    gau = [1/np.sqrt(2*np.pi*v[i]) * np.exp(-(x-m[i])**2/(v[i]*2)) for i in range(3)]
    g = [gau[i] * prior[i] for i in range(3)]
    rt = np.argmax(g) + 1
    return rt

# read input data
data = np.array(pd.read_csv("input_2.csv"))

# split input data
train_data_size = int(len(data) * 0.8)
train_data = data[0: train_data_size]
test_data = data[train_data_size:]
x_train, y_train = train_data[:, 0:1], train_data[:, 1].astype(int)
x_test, y_test = test_data[:, 0:1], test_data[:, 1].astype(int)

# number of training points
N = len(y_train)

# number of testing points
M = len(y_test)

# prior probability
prior_p = [sum(y_train == i + 1) / N for i in range(3)]
print("priors: ", prior_p)
# priors:  [0.34208333333333335, 0.3279166666666667, 0.33]

# maximum likelihood estimator
mean = [np.mean(x_train[y_train == i+1]) for i in range(3)]
variance = [np.var(x_train[y_train == i+1]) for i in range(3)]
print("mean: ", mean)
print("variance: ", variance)
# mean:  [-0.025056430115166065, 3.0494696695961974, -2.859953974797454]
# variance:  [1.0604933047619922, 3.9256875493805583, 9.706441924363155]

# predict test data using discriminant function
y_pre = np.array([0] * M)
for i in range(M):
    y_pre[i] = disfun(x_test[i], prior_p, mean, variance)

# confusion matrix
predict = pd.Series(y_pre, name="Predict")
actual = pd.Series(y_test, name="Actual")
con_matrix = pd.crosstab(actual, predict)
print(con_matrix)
# Predict    1    2    3
# Actual
# 1        194   11   15
# 2         42  162    2
# 3         37   13  124

diagcm = np.diag(con_matrix)

# accuracy
accuracy = sum(diagcm) / sum(np.sum(con_matrix))
print("accuracy: ", accuracy)
# accuracy: 0.8

# precision, recall, f1 score
precision = [0] * 3
recall = [0] * 3
f1_score = [0] * 3
for i in range(3):
   sum_col = sum(con_matrix[i+1])
   precision[i] = diagcm[i] / sum_col
   recall[i] = diagcm[i] / np.sum(con_matrix, axis=1)[i + 1]
   f1_score[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])

print("precision: ", precision)
# precision:  [0.7106227106227107, 0.8709677419354839, 0.8794326241134752]

print("recall: ", recall)
# recall:  [0.8818181818181818, 0.7864077669902912, 0.7126436781609196]

print("f1 score: ", f1_score)
# f1 score:  [0.7870182555780934, 0.826530612244898, 0.7873015873015873]

# average f1 score
average_f1_score = np.mean(f1_score)
print("average f1 score: ", average_f1_score)
# average f1 score:  0.8002834850415262

