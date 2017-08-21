import math
import csv
import numpy as np
import random
import pickle
import re
import itertools

# hacky path update to import peerselect
import sys
sys.path.insert(0, '/Users/akahng/Dropbox/RESEARCH/Procaccia/17-impartiality/peerselection-master/peerselect')
from peerselect import impartial_all as impartial
from peerselect import profile_generator
from peerselect import distance_helper as dist

'''
Super hacky; please make nicer later!
'''

# Mallows stuff here
def init_mallows(n, p):
	agents = np.arange(0,n)
	profile = profile_generator.generate_mallows_mixture_profile(agents, agents, [1.0], [agents], [p])

	cmps = []

	# make this profile into a cmps matrix
	for key in profile.keys():
		ranking = profile[key]
		n = len(ranking)
		for a in range(n):
			for b in range(a+1, n):
				cmps.append([key, ranking[a], ranking[b], 1])

	return cmps

def print_avgs(name, data):
	print(name)
	print('KT: {0}, FR: {1}, MD: {2}, CY: {3}, HM: {4}'.format(data[0], data[1], data[2], data[3], data[4]))
	print('')


def repeat(cmps, k, num_trials, voting_rule):
	comm_dists = []
	bip_dists = []
	kp_dists = []

	if voting_rule == impartial.kemeny:
		print('Using Kemeny')
		ref_sol = impartial.solve_kemeny(cmps)
	elif voting_rule == impartial.borda:
		print('Using Borda')
		ref_sol = impartial.solve_borda(cmps)
	else:
		sys.exit('Invalid voting rule.')

	for x in range(num_trials):
		comm_sol = impartial.committee_naive(cmps, k, voting_rule)
		bip_sol = impartial.bipartite(cmps, k, voting_rule)
		kp_sol = impartial.kpartite(cmps, k, voting_rule)

		# print(comm_sol)
		# print(bip_sol)
		# print(kp_sol)
		# print(ref_sol)

		comm_KT = dist.dKT(ref_sol, comm_sol)
		bip_KT = dist.dKT(ref_sol, bip_sol)
		kp_KT = dist.dKT(ref_sol, kp_sol)

		comm_FR = dist.dFR(ref_sol, comm_sol)
		bip_FR = dist.dFR(ref_sol, bip_sol)
		kp_FR = dist.dFR(ref_sol, kp_sol)

		comm_MD = dist.dMD(ref_sol, comm_sol)
		bip_MD = dist.dMD(ref_sol, bip_sol)
		kp_MD = dist.dMD(ref_sol, kp_sol)

		comm_CY = dist.dCY(ref_sol, comm_sol)
		bip_CY = dist.dCY(ref_sol, bip_sol)
		kp_CY = dist.dCY(ref_sol, kp_sol)

		comm_HM = dist.dHM(ref_sol, comm_sol)
		bip_HM = dist.dHM(ref_sol, bip_sol)
		kp_HM = dist.dHM(ref_sol, kp_sol)

		bip_dists.append((bip_KT, bip_FR, bip_MD, bip_CY, bip_HM))
		comm_dists.append((comm_KT, comm_FR, comm_MD, comm_CY, comm_HM))
		kp_dists.append((kp_KT, kp_FR, kp_MD, kp_CY, kp_HM))

		# comm_sol = impartial.committee_naive(cmps, k, impartial.borda)
		# bip_sol = impartial.bipartite(cmps, k, impartial.borda)
		# kp_sol = impartial.kpartite(cmps, k, impartial.borda)

	avg_comm_dists = list(np.mean(np.array(comm_dists), axis=0))
	avg_bip_dists = list(np.mean(np.array(bip_dists), axis=0))
	avg_kp_dists = list(np.mean(np.array(kp_dists), axis=0))

	print_avgs('comm', avg_comm_dists)
	print_avgs('bip', avg_bip_dists)
	print_avgs('kp', avg_kp_dists)

	# print(avg_comm_dists, avg_bip_dists, avg_kp_dists)

	print('comm: {0}, bip: {1}, kp: {2}'.format(comm_dists, bip_dists, kp_dists))

	return (comm_dists, bip_dists, kp_dists)


def run_all():
	test_n = [12]
	test_k = [3,4,6]
	test_p = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
	num_trials = 100

	k_results = {}
	b_results = {}

	for n,k,p in itertools.product(test_n, test_k, test_p):
		cmps = init_mallows(n,p)
		kemeny_results = repeat(cmps, k, num_trials, impartial.kemeny)
		borda_results = repeat(cmps, k, num_trials, impartial.borda)

		# append these things
		k_results[(n,k,p)] = kemeny_results
		b_results[(n,k,p)] = borda_results

	# save everything -- come up with a good indexing method!
	with open('./results/k_results.pickle', 'wb') as k_output_file:
		pickle.dump(k_results, k_output_file)
	with open('./results/b_results.pickle', 'wb') as b_output_file:
		pickle.dump(b_results, b_output_file)

def main():
	# repeat(exp_cmps1, k, num_trials)
	# kemeny_results = repeat(cmps, k, num_trials, impartial.kemeny)
	# borda_results = repeat(cmps, k, num_trials, impartial.borda)

	run_all()

if __name__ == '__main__':
	main()