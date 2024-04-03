import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = [[2,81], [4, 93], [6, 91], [8, 97]]
x = [i[0] for i in data]
y = [i[1] for i in data]

plt.figure(figsize=(8,5))
plt.scatter(x, y)
plt.show()

x_data = np.array(x)
y_data = np.array(y)

a = 0
b = 0

lr = 0.08  # 시작 학습률

epochs = 2001
prev_mse = 0

#경사 하강법 시작
for i in range(epochs):
  y_pred = a * x_data +b
  error = y_data - y_pred
  mse = np.mean((y_pred - y_data) ** 2)

  a_diff = -(1/len(x_data)) * sum(x_data * (error))
  b_diff = -(1/len(x_data)) * sum(y_data - y_pred)

  if (mse > prev_mse):
    if (prev_mse == 0):
      prev_mse = mse
    else:
      lr = lr - 0.01
  else:
    prev_mse = mse

  if i % 400 == 0:
    print("epoch=%.f, 기울기=%.04f, 절편=%.04f" % (i, a, b))
    print("LR: %.04f, a_diff: %.02f, b_diff: %.02f, MSE:%.10f" % (lr, a_diff, b_diff, mse))

  a = a - lr * a_diff
  b = b - lr * b_diff

y_pred = a * x_data + b
plt.scatter(x, y, marker='x')

# 오차 거리
plt.plot([min(x_data), max(x_data)], [min(y_pred), max(y_pred)])
plt.plot([x, (x_data)], [y, (y_pred)])

plt.show()
