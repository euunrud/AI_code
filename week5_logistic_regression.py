import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#실습 4_1
#공부시간 X와 성적 Y의 리스트
data = [[2.0, 0.0], [4.0, 0.0], [6.0, 0.0], [8.0, 1.0], [10.0, 1.0], [12.0, 1.0], [14.0, 1.0]]
a = 0
b = 0

#학습률
lr = 0.05

x = [i[0] for i in data]
y = [i[1] for i in data]

x_data = np.array(x)
y_data = np.array(y)
loss = []

#시그모이드 함수를 정의
def sigmoid(x):
    return 1 / (1 + np.e ** (-x))

def error_sum():
  return ((y_data - sigmoid(a*x_data + b))**2).mean()

#경사 하강법을 실행
for i in range(10001):
      a_diff = (1/len(x_data))*sum(x_data *(sigmoid(a*x_data + b) - y_data))
      b_diff = (1/len(x_data))*sum(sigmoid(a*x_data+b)-y_data)
      a = a - lr * a_diff
      b = b - lr * b_diff

      loss.append(error_sum())

      if i % 1000 == 0:    # 1000번 반복될 때마다 각 x_data값에 대한 현재의 a값, b값을 출력
          print('epoch=%.f. 기울기=%.04f. 절편=%.04f' % (i, a, b))
          print('Error value: %.04f' % loss[i])

# 기울기와 절편을 이용해 그래프
x_range = (np.arange(0, 10001, 1)) # 그래프로 나타낼 x 범위
plt.plot(np.arange(0, 10001, 1), loss)
plt.show()

for i in range(len(x_data)):
    print("Input: %.1f.  Real: %.1f.  Expect: %.4f" % (x_data[i], y_data[i], sigmoid(a * x_data[i] + b)))
