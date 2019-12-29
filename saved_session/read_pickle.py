import pickle

with open('electric_mkt_data.pkl', 'rb') as f:
	data_new = pickle.load(f)

print (data_new)