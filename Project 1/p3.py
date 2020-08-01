import pandas as pd
import numpy as np

# multi-features
# x1: feature value 1 (Bernoulli distribution)
# x2: feature value 2 (Gaussian distribution)
# y: class


def disfun(x1, x2, prior, bino, m, v):
    p = [bino[i] if x1 == 1 else 1-bino[i] for i in range(3)]
    gau = [1/np.sqrt(2*np.pi*v[i]) * np.exp(-(x2-m[i])**2/(v[i]*2)) for i in range(3)]
    g = [p[i] * gau[i] * prior[i] for i in range(3)]
    rt = np.argmax(g) + 1
    return rt


# read input data
data = np.array(pd.read_csv("input_3.csv"))

# split input data
train_data_size = int(len(data) * 0.8)
train_data = data[0: train_data_size]
test_data = data[train_data_size:]
x_train1, x_train2, y_train = train_data[:, 0:1], train_data[:, 1:2], train_data[:, 2].astype(int)
x_test1, x_test2, y_test = test_data[:, 0:1], test_data[:, 1:2], test_data[:, 2].astype(int)

# number of training points
N = len(y_train)

# number of testing points
M = len(y_test)

# prior probability
prior_p = [sum(y_train == i + 1) / N for i in range(3)]
print("priors: ", prior_p)
# priors:  [0.32916666666666666, 0.3333333333333333, 0.3375]

# maximum likelihood estimator
p = [0] * 3
for i in range(3):
   sum_data = x_train1[y_train == i+1]
   p[i] = sum(sum_data[:, 0]) / len(sum_data)
mean = [np.mean(x_train2[y_train == i+1]) for i in range(3)]
variance = [np.var(x_train2[y_train == i+1]) for i in range(3)]
print("estimated p: ", p)
print("mean: ", mean)
print("variance: ", variance)
# estimated p:  [0.1, 0.515, 0.7716049382716049]
# mean:  [1.0246692865939961, 4.969530068401736, 9.888599469860715]
# variance:  [0.24928441068852913, 6.969167050348478, 20.618919124235582]

# predict test data using discriminant function
y_pre = np.array([0] * M)
for i in range(M):
    y_pre[i] = disfun(x_test1[i], x_test2[i], prior_p, p, mean, variance)
# print(y_pre)

# confusion matrix
con_matrix = [[0] * 3 for i in range(3)]
for c in range(M):
   con_matrix[y_test[c]-1][y_pre[c]-1] += 1
con_matrix = np.reshape(con_matrix, (3, 3))
index = [i for i in range(1, 4)]
con_matrix = pd.DataFrame(con_matrix, columns=index, index=index)
print(con_matrix)
#      1    2    3
# 1  204    9    0
# 2   15  144   27
# 3    4   50  147

diagcm = np.diag(con_matrix)

# accuracy
accuracy = sum(diagcm) / sum(np.sum(con_matrix))
print("accuracy: ", accuracy)
# accuracy: 0.825

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
# precision:  [0.9147982062780269, 0.7093596059113301, 0.8448275862068966]

print("recall: ", recall)
# recall:  [0.9577464788732394, 0.7741935483870968, 0.7313432835820896]

print("f1 score: ", f1_score)
# f1 score:  [0.9357798165137613, 0.7403598971722365, 0.784]

# average f1 score
average_f1_score = np.mean(f1_score)
print("average f1 score: ", average_f1_score)
# average f1 score:  0.8200465712286661


