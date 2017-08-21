import math
import csv
import numpy as np
import random
# import matplotlib
# import matplotlib.pyplot as plt
import pickle
import re

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
p = 0.1
agents = np.arange(0,n)
profile = profile_generator.generate_mallows_mixture_profile(agents, agents, [1.0], [agents], [p])

# load input file and set up all preliminaries
exp_cmps1 = None
with open('/Users/akahng/Dropbox/RESEARCH/Procaccia/17-impartiality/final-matrix-s2-for-d-and-a.csv', 'r') as infile:
    lines = infile.readlines()
    lines = [line.strip().split(",") for line in lines]
    lines = [[re.sub('[a-z]', '', line[0]), re.sub('[a-z]', '', line[1]), re.sub('[a-z]', '', line[2]), int(line[3])] for line in lines]
    exp_cmps1 = lines

k = 6

# kem_sol = impartial.solve_k(exp_cmps1, k)
# topk_sol = impartial.topk(exp_cmps1, k)
# comm_sol = impartial.committee_naive(exp_cmps1, k)
# bip_sol = impartial.bipartite(exp_cmps1, k)
# kp_sol = impartial.kpartite(exp_cmps1, k)

def overlap(a,b):
	overlap_list = list(set(a).intersection(set(b)))
	return (len(overlap_list), len(overlap_list) / len(a))

# print('kemeny:', kem_sol, overlap(kem_sol, kem_sol))
# print('topk:', topk_sol, overlap(kem_sol, topk_sol))
# print('committee:', comm_sol, overlap(kem_sol, comm_sol))
# print('bipartite:', bip_sol, overlap(kem_sol, bip_sol))
# print('kpartite:', kp_sol, overlap(kem_sol, kp_sol))


num_trials = 30


def repeat(cmps, k, num_trials):
	topk_overlap = []
	comm_overlap = []
	bip_overlap = []
	kp_overlap = []

	kem_sol = impartial.solve_k(cmps, k)

	for x in range(num_trials):
		topk_sol = impartial.topk(cmps, k, impartial.kemeny)
		comm_sol = impartial.committee_naive(cmps, k, impartial.kemeny)
		bip_sol = impartial.bipartite(cmps, k, impartial.kemeny)
		kp_sol = impartial.kpartite(cmps, k, impartial.kemeny)

		topk_overlap.append(overlap(kem_sol, topk_sol)[0])
		comm_overlap.append(overlap(kem_sol, comm_sol)[0])
		bip_overlap.append(overlap(kem_sol, bip_sol)[0])
		kp_overlap.append(overlap(kem_sol, kp_sol)[0])

		topk_frac = sum(topk_overlap) / (k * len(topk_overlap))
		comm_frac = sum(comm_overlap) / (k * len(comm_overlap))
		bip_frac = sum(bip_overlap) / (k * len(bip_overlap))
		kp_frac = sum(kp_overlap) / (k * len(kp_overlap))

	print('topk: {0}, comm: {1}, bip: {2}, kp: {3}'.format(topk_frac, comm_frac, bip_frac, kp_frac))

def main():
	repeat(exp_cmps1, k, num_trials)

if __name__ == '__main__':
	main()