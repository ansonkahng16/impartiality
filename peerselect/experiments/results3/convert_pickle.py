import pickle
import sys
import csv

base = pickle.load(open('topk_versus_base.pickle', 'rb'))
vanilla = pickle.load(open('topk_versus_vanilla.pickle', 'rb'))

base_data = {}
vanilla_data = {}

for x in base: 
	# x is, e.g., (130, 25, 9, 5, 0.0, 100, 'DollarPartitionRaffle')
	tmp_list = [i/x[1] for i in base[x]]
	tmp_avg = sum(tmp_list) / len(tmp_list)
	base_data[x] = tmp_avg

for x in vanilla: 
	# x is, e.g., (130, 25, 9, 5, 0.0, 100, 'DollarPartitionRaffle')
	tmp_list = [i/x[1] for i in vanilla[x]]
	tmp_avg = sum(tmp_list) / len(tmp_list)
	vanilla_data[x] = tmp_avg

with open('topk_versus_base.txt', 'w') as base_out:
	base_file = csv.writer(base_out)
	for row in base:
		tmp = row + (base_data[row],)
		base_file.writerow(tmp)

with open('topk_versus_vanilla.txt', 'w') as vanilla_out:
	vanilla_file = csv.writer(vanilla_out)
	for row in vanilla:
		tmp = row + (vanilla_data[row],)
		vanilla_file.writerow(tmp)

