import numpy as np

input_data = np.array([[2, 3], [5, 1]])
print(input_data)

x = input_data.reshape(-1)

w1 = np.array([2, 1, -3, 3])
w2 = np.array([1, -3, 1, 3])
b1 = 3
b2 = 3

W = np.array([w1, w2])
b = np.array([b1, b2])

weight_sum = np.dot(W, x) + b
print(weight_sum)

res = 1 / (1 + np.exp(-weight_sum))
print(res)
