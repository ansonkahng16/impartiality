import pickle
import numpy as np
import sys
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties

vanilla = pickle.load(open('long_run_with_bucket_versus_base.pickle', 'rb'))  # not actually vanilla; this is base
# vanilla = pickle.load(open('long_run_with_bucket_versus_vanilla.pickle', 'rb'))

# print(base)
# print(vanilla)

vanilla_data = {}

for x in vanilla:  
	# x is, e.g., (130, 25, 9, 5, 0.0, 100, 'DollarPartitionRaffle')
	tmp_list = [i/x[1] for i in vanilla[x]]
	tmp_avg = sum(tmp_list) / len(tmp_list)
	vanilla_data[x] = tmp_avg

# top row in paper
test_m = [5, 13, 40, 60]
test_k = [10, 20, 30]
test_l = [3, 4, 5]
test_p = [0.0, 0.2, 0.5]
mechanisms = ["Vanilla", "ExactDollarPartition", "Partition", "DollarPartitionRaffle", "DollarRaffle", "CredibleSubset", "Bucket"]

'''
Notes about structure
x = key: (120, 30, 5, 4, 0.0, 100, 'DollarRaffle')
x[0] = 120
x[1] = k: [10, 20, 30]
x[2] = m: [5, 13, 40, 60]
x[3] = l: [3, 4, 5]
x[4] = p: [0.0, 0.2, 0.5]
x[5] = 100
x[6] = mechanism 

dict[x] = avg performance
'''

data_dict = {}


# load and format data
for l in test_l:
	for p in test_p:
		# with a fixed l and p
		for m in test_m:
			for k in test_k:
				# print('l: {}, p: {}, m: {}, k: {}'.format(l, p, m, k))
				for x in vanilla_data:
					if x[3] == l and x[4] == p and x[2] == m and x[1] == k:
						data_dict[(k,m,l,p)] = (mechanisms.index(x[6]), vanilla_data[x])


# vary k
for l in test_l:
	for p in test_p:
		for m in test_m:  # with a fixed l, p, m
			tmp = [[],[],[],[],[],[],[]]
			for k in test_k:
				# print('l: {}, p: {}, m: {}, k: {}'.format(l, p, m, k))
				for x in vanilla_data:
					if x[3] == l and x[4] == p and x[2] == m and x[1] == k:
						tmp[mechanisms.index(x[6])].append(vanilla_data[x])
			fig, ax = plt.subplots()

			ind = np.arange(len(test_k))
			width = 0.1

			v_plot = ax.bar(ind, tmp[0], width, color='blue')
			edp_plot = ax.bar(ind + width, tmp[1], width, color='green')
			p_plot = ax.bar(ind + 2*width, tmp[2], width, color='red')
			dpr_plot = ax.bar(ind + 3*width, tmp[3], width, color='orange')
			dr_plot = ax.bar(ind + 4*width, tmp[4], width, color='purple')
			cs_plot = ax.bar(ind + 5*width, tmp[5], width, color='brown')
			b_plot = ax.bar(ind + 6*width, tmp[6], width, color='cyan')

			ax.set_title('Varying k: (l: {}, p: {}, m: {})'.format(l, p, m))
			ax.set_ylabel('Accuracy')
			ax.set_xlabel('k')
			ax.set_xticks(ind + width / 2)
			ax.set_xticklabels(tuple([str(x) for x in test_k]))

			fontP = FontProperties()
			fontP.set_size('small')

			ax.legend((v_plot[0], edp_plot[0], p_plot[0], dpr_plot[0], dr_plot[0], cs_plot[0], b_plot[0]), ("Vanilla", "ExactDollarPartition", "Partition", "DollarPartitionRaffle", "DollarRaffle", "CredibleSubset", "Bucket"), prop=fontP, bbox_to_anchor=(1.25,1.05), ncol=1)

			filename = 'k_figs_base/k_l={}_p={}_m={}.png'.format(l,int(p*10),m)
			print(filename)

			plt.savefig(filename)


# vary m
for l in test_l:
	for p in test_p:
		for k in test_k:  # with a fixed l, p, k
			tmp = [[],[],[],[],[],[],[]]
			for m in test_m:
				# print('l: {}, p: {}, m: {}, k: {}'.format(l, p, m, k))
				for x in vanilla_data:
					if x[3] == l and x[4] == p and x[2] == m and x[1] == k:
						tmp[mechanisms.index(x[6])].append(vanilla_data[x])
			fig, ax = plt.subplots()

			ind = np.arange(len(test_m))
			width = 0.1

			v_plot = ax.bar(ind, tmp[0], width, color='blue')
			edp_plot = ax.bar(ind + width, tmp[1], width, color='green')
			p_plot = ax.bar(ind + 2*width, tmp[2], width, color='red')
			dpr_plot = ax.bar(ind + 3*width, tmp[3], width, color='orange')
			dr_plot = ax.bar(ind + 4*width, tmp[4], width, color='purple')
			cs_plot = ax.bar(ind + 5*width, tmp[5], width, color='brown')
			b_plot = ax.bar(ind + 6*width, tmp[6], width, color='cyan')

			ax.set_title('Varying m: (l: {}, p: {}, k: {})'.format(l, p, k))
			ax.set_ylabel('Accuracy')
			ax.set_xlabel('m')
			ax.set_xticks(ind + width / 2)
			ax.set_xticklabels(tuple([str(x) for x in test_m]))

			fontP = FontProperties()
			fontP.set_size('small')

			ax.legend((v_plot[0], edp_plot[0], p_plot[0], dpr_plot[0], dr_plot[0], cs_plot[0], b_plot[0]), ("Vanilla", "ExactDollarPartition", "Partition", "DollarPartitionRaffle", "DollarRaffle", "CredibleSubset", "Bucket"), prop=fontP, bbox_to_anchor=(1.25,1.05), ncol=1)

			filename = 'm_figs_base/m_l={}_p={}_k={}.png'.format(l,int(p*10),k)
			print(filename)

			plt.savefig(filename)
