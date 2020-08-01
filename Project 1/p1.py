import pandas as pd
import numpy as np

# multinomial distribution
# x: feature value
# y: class


def disfun(x_test, prior_p, p):
   p_i = [p[i] if x_test == 1 else 1-p[i] for i in range(3)]
   g = [p_i[i] * prior_p[i] for i in range(3)]
   rt = np.argmax(g) + 1
   return rt

# read input data
data = np.array(pd.read_csv("input_1.csv"))

# split input data
train_data_size = int(len(data) * 0.8)
train_data = data[0: train_data_size]
test_data = data[train_data_size:]
x_train, y_train = train_data[:, 0:1], train_data[:, 1]
x_test, y_test = test_data[:, 0:1], test_data[:, 1]
# (2400, 2)
# print("train data shape: ", train_data.shape)

# (2400, 1)
# print("x_train shape: ", x_train.shape)

# (2400, )
# print("y_train shape: ", y_train.shape)

# number of training points
N = len(y_train)

# number of testing points
M = len(y_test)

# prior probability
prior_p = [sum(y_train == i+1)/N for i in range(3)]
print("priors: ", prior_p)
# priors: [0.34125, 0.3383333333333333, 0.3204166666666667]

# maximum likelihood estimator
p = [0] * 3
for i in range(3):
   sum_data = x_train[y_train == i+1]
   p[i] = sum(sum_data[:, 0]) / len(sum_data)
print("estimate p:", p)
# estimate
# p1: 0.344322
# estimate
# p2: 0.612069
# estimate
# p3: 0.401821

# predict test data using discriminant function
y_pre = np.array([0] * M)
for i in range(M):
   y_pre[i] = disfun(x_test[i], prior_p, p)

# confusion matrix
con_matrix = [[0] * 3 for i in range(3)]
for c in range(M):
   con_matrix[y_test[c]-1][y_pre[c]-1] += 1
con_matrix = np.reshape(con_matrix, (3, 3))
index = [i for i in range(1, 4)]
con_matrix = pd.DataFrame(con_matrix, columns=index, index=index)
print("confusion matrix: \n", con_matrix)
# predict  1    2  3
# actual
# 1       135   70  0
# 2       70  108  0
# 3       148   69  0

diagcm = np.diag(con_matrix)
# print("diag: ", diagcm)
# print("np.sum confusion matrix: \n", np.sum(con_matrix))
# np.sum confusion matrix:
# 1    353
# 2    247
# 3      0
# dtype: int64

# accuracy
accuracy = sum(diagcm) / sum(np.sum(con_matrix))
print("accuracy: ", accuracy)
# accuracy:  0.405

# precision, recall, f1 score
precision = [0] * 3
recall = [0] * 3
f1_score = [0] * 3
for i in range(3):
   sum_col = sum(con_matrix[i+1])
   precision[i] = 0 if sum_col == 0 else diagcm[i] / sum_col
   recall[i] = diagcm[i] / np.sum(con_matrix, axis=1)[i + 1]
   f1_score[i] = 0 if precision[i] + recall[i] == 0 else 2 * precision[i] * recall[i] / (precision[i] + recall[i])

print("precision: ", precision)
# precision:  [0.38243626062322944, 0.43724696356275305, 0]

print("recall: ", recall)
# recall:  [0.6585365853658537, 0.6067415730337079, 0.0]

print("f1 score: ", f1_score)
# f1 score:  [0.4838709677419354, 0.5082352941176471, 0]

# average f1 score
average_f1_score = np.mean(f1_score)
print("average f1 score: ", average_f1_score)
# average f1 score:  0.3307020872865275






