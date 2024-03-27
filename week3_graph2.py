import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('/content/drive/MyDrive/colab/deeplearning/dataset/pima-indians-diabetes.csv',
                 names = ["pregnant", "plasma", "pressure", "thickness", "insulin", "BMI", "pedigree", "age", "class"])

aver = df.describe().age[1]
print('Average age:', round(aver, 2))
data = df[['age']].values

low_age = df[df['age'] <= aver].thickness
high_age = df[df['age'] >= aver].thickness
low_age.plot.hist(bins=100, alpha=0.5)
high_age.plot.hist(bins=100, alpha=0.5)

plt.legend(['low_age', 'high_age'])
plt.show()