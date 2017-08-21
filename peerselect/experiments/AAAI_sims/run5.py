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
sys.path.insert(0, '/afs/andrew.cmu.edu/usr20/akahng/impartiality/peerselect')
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

def dKEM(ranking, cmps):  # reverse Kemeny score -- counts # agreements
	mapping, mtx = impartial.convert(cmps)

	score = 0
	for i in range(len(ranking)):
		for j in range(i, len(ranking)):
			x = ranking[i]  # first index >
			y = ranking[j]  # second index
			score += mtx[x][y]  # (don't) reverse them

	return score

def repeat(cmps, k, num_trials, voting_rule):
	comm_dists = []
	bip_dists = []
	kp_dists = []

	if voting_rule == impartial.kemeny:
		print('Using Kemeny')
		ref_sol = impartial.solve_kemeny(cmps)
		ref_KEM = dKEM(ref_sol, cmps)
		print('ref sol dist: {}'.format(ref_KEM))
	elif voting_rule == impartial.borda:
		sys.exit('No more Borda')
		
		print('Using Borda')
		ref_sol = impartial.solve_borda(cmps)
	else:
		sys.exit('Invalid voting rule.')

	for x in range(num_trials):
		comm_sol = impartial.committee_naive(cmps, k, voting_rule)
		bip_sol = impartial.bipartite(cmps, k, voting_rule)
		kp_sol = impartial.kpartite(cmps, k, voting_rule)

		comm_KEM = dKEM(comm_sol, cmps) / ref_KEM
		bip_KEM = dKEM(bip_sol, cmps) / ref_KEM
		kp_KEM = dKEM(kp_sol, cmps) / ref_KEM

		# print('{}, {}, {}'.format(comm_KEM, bip_KEM, kp_KEM))

		bip_dists.append(bip_KEM)
		comm_dists.append(comm_KEM)
		kp_dists.append(kp_KEM)

	avg_comm_dist = np.mean(comm_dists)
	avg_bip_dist = np.mean(bip_dists)
	avg_kp_dist = np.mean(kp_dists)

	# print('comm: {0}, bip: {1}, kp: {2}'.format(comm_dists, bip_dists, kp_dists))

	# sys.exit('break early')

	return (comm_dists, bip_dists, kp_dists)


def run_all():
	test_n = [8, 12, 16]
	test_p = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
	# test_p = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
	const = 4
	num_trials = 1

	k_results = {}
	b_results = {}

	for n,p in itertools.product(test_n, test_p):
		print(n,p)
		cmps = init_mallows(n,p)
		kemeny_results = repeat(cmps, int(n/const), num_trials, impartial.kemeny)

		# append these things
		k_results[(n,p)] = kemeny_results

		# save everything each run!
		with open('./results5/k_results_{}_{}.pickle'.format(n, '{num:02d}'.format(num=int(p*10))), 'wb') as k_output_file:
			pickle.dump(k_results, k_output_file)

def main():
	# repeat(exp_cmps1, k, num_trials)
	# kemeny_results = repeat(cmps, k, num_trials, impartial.kemeny)
	# borda_results = repeat(cmps, k, num_trials, impartial.borda)

	run_all()

if __name__ == '__main__':
	main()