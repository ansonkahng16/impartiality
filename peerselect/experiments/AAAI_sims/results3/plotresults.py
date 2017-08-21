import pickle
import numpy as np
import sys
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties

k_data = pickle.load(open('k_results.pickle', 'rb'))

'''
data format: 
	{(n, k, p): ([(dKT, dFR, dMD, dCY, dHM), ..., (dKT, dFR, dMD, dCY, dHM)])}
	([comm_dists, bip_dists, kp_dists])
'''

test_n = [16, 24, 32]
test_p = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
# k is a fraction (1/8) of n

num_algs = 3  # comm, bip, kp
num_dist_funs = 1  # KEM 

def plotdata(data):
	for p in test_p:
			comm_data = []
			bip_data = []
			kp_data = []
			
			for x in data:
				# load data for all values of phi
				for n in test_n:
					if x[0] == n and x[1] == p:
						comm_data.append(data[x][0])
						bip_data.append(data[x][1])
						kp_data.append(data[x][2])

			for i in range(num_algs):
				# i = 0: comm
				# i = 1: bip
				# i = 2: kp
				if i == 0:
					title = 'committee'
					all_dists = comm_data
				elif i == 1:
					title = 'bipartite'
					all_dists = bip_data
				elif i == 2:
					title = 'kpartite'
					all_dists = kp_data
				else:
					sys.exit('Invalid')

				fig, ax = plt.subplots()

				plot_tmp = ax.boxplot(all_dists)

				title_str = '{}_p={}'.format(title,'{num:02d}'.format(num=int(p*10)))

				ax.set_title(title_str)
				ax.set_xlabel('n')
				ax.set_ylabel('Distance')

				data_labels = [str(n) for n in test_n]

				xtickNames = plt.setp(ax, xticklabels=data_labels)
				plt.setp(xtickNames, rotation=45, fontsize=6)

				plot_dir = 'plots'
				save_title = '{}/{}.png'.format(plot_dir, title_str)

				print(save_title)

				plt.savefig(save_title)

plotdata(k_data)
