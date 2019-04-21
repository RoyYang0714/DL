import pandas as pd

def spilt():
	df = pd.read_csv('data/Data.csv')
	df['dws'] = 0.
	tmp = df['dws'].copy()
	tmp[df.Activities_Types == 1] = 1.
	df['dws'] = tmp.copy()
	df['ups'] = 0.
	tmp = df['ups'].copy()
	tmp[df.Activities_Types == 2] = 1.
	df['ups'] = tmp.copy()
	df['sit'] = 0.
	tmp = df['sit'].copy()
	tmp[df.Activities_Types == 3] = 1.
	df['sit'] = tmp.copy()
	df['std'] = 0.
	tmp = df['std'].copy()
	tmp[df.Activities_Types == 4] = 1.
	df['std'] = tmp.copy()
	df['wlk'] = 0.
	tmp = df['wlk'].copy()
	tmp[df.Activities_Types == 5] = 1.
	df['wlk'] = tmp.copy()
	df['jog'] = 0.
	tmp = df['jog'].copy()
	tmp[df.Activities_Types == 6] = 1.
	df['jog'] = tmp.copy()

	df = df.drop(['Activities_Types'], axis=1)
	data_train = df.sample(frac=0.8).reset_index(drop=True)
	data_valid = df.sample(frac=0.2).reset_index(drop=True)

	return data_train, data_valid