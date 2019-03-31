import pandas as pd

data = pd.read_csv('./data/train.csv')
df = pd.get_dummies(data)
df['label'] = 0
df.label[df.G3>=10] = 1

data_train = df.sample(frac=0.8).reset_index(drop=True)
data_test = df.sample(frac=0.2).reset_index(drop=True)
data_train.to_csv('./data/training.csv', index=False)
data_test.to_csv('./data/test.csv', index=False)

data = pd.read_csv('./data/test_no_G3.csv')
df = pd.get_dummies(data)
df.to_csv('./data/test_no_G3.csv', index=False)