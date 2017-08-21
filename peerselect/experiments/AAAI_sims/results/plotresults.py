import pickle
import numpy as np
import sys
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties

k_data = pickle.load(open('k_results.pickle', 'rb'))
b_data = pickle.load(open('b_results.pickle', 'rb'))

'''
data format: 
	{(n, k, p): ([(dKT, dFR, dMD, dCY, dHM), ..., (dKT, dFR, dMD, dCY, dHM)])}
	([comm_dists, bip_dists, kp_dists])
'''

test_n = [12]
test_k = [3,4,6]
test_p = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

num_algs = 3  # comm, bip, kp
num_dist_funs = 5  # KT, FR, MD, CY, HM


def plotdata(data):
	for n in test_n:
		for k in test_k:
			for p in test_p:
				for x in data:
					if x[0] == n and x[1] == k and x[2] == p:
						comm_dists = data[x][0]
						bip_dists = data[x][1]
						kp_dists = data[x][2]
						
						comm_dists_dKT = [c[0] for c in comm_dists]
						comm_dists_dFR = [c[1] for c in comm_dists]
						comm_dists_dMD = [c[2] for c in comm_dists]
						comm_dists_dCY = [c[3] for c in comm_dists]
						comm_dists_dHM = [c[4] for c in comm_dists]

						bip_dists_dKT = [c[0] for c in bip_dists]
						bip_dists_dFR = [c[1] for c in bip_dists]
						bip_dists_dMD = [c[2] for c in bip_dists]
						bip_dists_dCY = [c[3] for c in bip_dists]
						bip_dists_dHM = [c[4] for c in bip_dists]

						kp_dists_dKT = [c[0] for c in kp_dists]
						kp_dists_dFR = [c[1] for c in kp_dists]
						kp_dists_dMD = [c[2] for c in kp_dists]
						kp_dists_dCY = [c[3] for c in kp_dists]
						kp_dists_dHM = [c[4] for c in kp_dists]

						comm_dists = [comm_dists_dKT, comm_dists_dFR, comm_dists_dMD, comm_dists_dCY, comm_dists_dHM]

						bip_dists = [bip_dists_dKT, bip_dists_dFR, bip_dists_dMD, bip_dists_dCY, bip_dists_dHM]

						kp_dists = [kp_dists_dKT, kp_dists_dFR, kp_dists_dMD, kp_dists_dCY, kp_dists_dHM]

						all_dists = comm_dists + bip_dists + kp_dists

						fig, ax = plt.subplots()

						ind = np.arange(num_algs * num_dist_funs + (num_algs - 1))
						width = 0.1

						plot_tmp = ax.boxplot(all_dists)


						title_str = 'n={}_k={}_p={}'.format(n,k,'{num:02d}'.format(num=int(p*10)))

						ax.set_title('' + title_str)
						ax.set_xlabel('')
						ax.set_ylabel('Distance')

						data_labels = ['Comm KT', 'Comm FR', 'Comm MD', 'Comm CY', 'Comm HM', 'Bip KT', 'Bip FR', 'Bip MD', 'Bip CY', 'Bip HM', 'Kp KT', 'Kp FR', 'Kp MD', 'Kp CY', 'Kp HM']

						xtickNames = plt.setp(ax, xticklabels=data_labels)
						plt.setp(xtickNames, rotation=45, fontsize=6)

						plot_dir = 'plots'
						save_title = '{}/{}.png'.format(plot_dir, title_str)

						print(save_title)

						plt.savefig(save_title)

def plotdata_bydistfun(data, title):
	for n in test_n:
		for k in test_k:
			for p in test_p:
				for x in data:
					if x[0] == n and x[1] == k and x[2] == p:
						comm_dists = data[x][0]
						bip_dists = data[x][1]
						kp_dists = data[x][2]
						
						comm_dists_dKT = [c[0] for c in comm_dists]
						comm_dists_dFR = [c[1] for c in comm_dists]
						comm_dists_dMD = [c[2] for c in comm_dists]
						comm_dists_dCY = [c[3] for c in comm_dists]
						comm_dists_dHM = [c[4] for c in comm_dists]

						bip_dists_dKT = [c[0] for c in bip_dists]
						bip_dists_dFR = [c[1] for c in bip_dists]
						bip_dists_dMD = [c[2] for c in bip_dists]
						bip_dists_dCY = [c[3] for c in bip_dists]
						bip_dists_dHM = [c[4] for c in bip_dists]

						kp_dists_dKT = [c[0] for c in kp_dists]
						kp_dists_dFR = [c[1] for c in kp_dists]
						kp_dists_dMD = [c[2] for c in kp_dists]
						kp_dists_dCY = [c[3] for c in kp_dists]
						kp_dists_dHM = [c[4] for c in kp_dists]

						KT_dists = [comm_dists_dKT, bip_dists_dKT, kp_dists_dKT]
						FR_dists = [comm_dists_dFR, bip_dists_dFR, kp_dists_dFR]
						MD_dists = [comm_dists_dMD, bip_dists_dMD, kp_dists_dMD]
						CY_dists = [comm_dists_dCY, bip_dists_dCY, kp_dists_dCY]
						HM_dists = [comm_dists_dHM, bip_dists_dHM, kp_dists_dHM]

						all_dists = KT_dists + FR_dists + MD_dists + CY_dists + HM_dists

						fig, ax = plt.subplots()

						ind = np.arange(num_algs * num_dist_funs + (num_algs - 1))
						width = 0.1

						plot_tmp = ax.boxplot(all_dists)


						title_str = 'n={}_k={}_p={}'.format(n,k,'{num:02d}'.format(num=int(p*10)))

						ax.set_title('' + title_str)
						ax.set_xlabel('')
						ax.set_ylabel('Distance')

						data_labels = ['Comm KT', 'Bip KT', 'Kp KT', 'Comm FR', 'Bip FR', 'Kp FR', 'Comm MD', 'Bip MD', 'Kp MD', 'Comm CY', 'Bip CY', 'Kp CY', 'Comm HM', 'Bip HM', 'Kp HM']

						xtickNames = plt.setp(ax, xticklabels=data_labels)
						plt.setp(xtickNames, rotation=45, fontsize=6)

						plot_dir = 'plots_{}'.format(title)
						save_title = '{}/{}.png'.format(plot_dir, title_str)

						print(save_title)

						plt.savefig(save_title)



# plotdata(k_data)

plotdata_bydistfun(k_data, 'kemeny')

plotdata_bydistfun(k_data, 'borda')
