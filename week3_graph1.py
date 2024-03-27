import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('/content/drive/MyDrive/colab/deeplearning/dataset/pima-indians-diabetes.csv',
                 names = ["pregnant", "plasma", "pressure", "thickness", "insulin", "BMI", "pedigree", "age", "class"])

df2 = df[['pregnant', 'plasma']]
data1 = df2.values
data2 = data1[:5]
print(data2)
print(type(data2))

x1 = []
y1 = []
for i in range(5):
    x1.append(data2[i][0])
for i in range(5):
    y1.append(data2[i][1])

plt.plot(x1, y1,color='lightGreen')
plt.plot(x1, y1,"o", color='green')