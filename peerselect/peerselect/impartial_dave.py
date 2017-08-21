import math
import itertools
import random
import copy
import numpy as np
from scipy import stats
import sys
from birkhoff import birkhoff_von_neumann_decomposition

## Some Settings
_DEBUG = False

def validate_matrix(score_matrix, partition={}):
	"""
	Validate and enforce constraints on a score matrix object:
	(1) must be square
	(2) must not score self
	Optional:
	(3) must not score others in my partition.
	(4) partition must not overlap.

	Parameters
	-----------
	score_matrix: array like
		The numerical scores of the agents for all the other agents.
		We use the convention that a[i,j] is the grade that agent 
		j gives to i.  This means that column j is all the grades
		*given* by agent j and that row i is the grades *recieved*
		by agent i.

	partition: dict
		non-overlapping partition of elements that are grouped together.

	Returns
	-----------
	score_matrix: array like
		score matrix obeying the above constraints.

	Notes
	-----------
	"""
	if score_matrix.ndim != 2 or score_matrix.shape[0] != score_matrix.shape[1]:
			print("score matrix is not square or has no values")
			return 0
	
	#Enforce the Diagonal is 0's
	for i in range(score_matrix.shape[0]):
			score_matrix[i,i] = 0.

	if partition != {}:
		if _DEBUG: print("\npartitions:\n" + str(partition))
		#Validate the partitions and check that 0's for the elements in my
		#partition.
		#Ensure that the partitions don't overlap.
		agent_set = list(itertools.chain(*partition.values()))
		if len(agent_set) != len(set(agent_set)):
		 print("partitioning contains duplicates")
		 return 0

		# Note we have to blank out the scores of the other agents in our partitions
		# before we do anything else.  We know we already have a 0. grade for ourselves.
		for c in partition.values():
			for pair in itertools.permutations(c, 2):
				score_matrix[pair[0],pair[1]] = 0.

	if _DEBUG: print("score matrix:\n"+str(score_matrix))

	return score_matrix

def normalize_score_matrix(score_matrix, partition={}):
	"""
	Normalize a score_matrix so that all the scores given 
	by each agent to other agents outside their clusters
	is equal to exactly 1.
	

	Parameters
	-----------
	score_matrix: array like
		The numerical scores of the agents for all the other agents.
		We use the convention that a[i,j] is the grade that agent 
		j gives to i.  This means that column j is all the grades
		*given* by agent j and that row i is the grades *recieved*
		by agent i.

	partition: dict
		A mapping from an integer --> list(indicies) where the list of 
		inidicies are the rows of the score_matrix that contain 
		the scores. 

	Returns
	-----------
	score_matrix: array like
		The numerical scores of the agents for all the other agents
		normalized such that each column sums to 1.
		We use the convention that a[i,j] is the grade that agent 
		j gives to i.  This means that column j is all the grades
		*given* by agent j and that row i is the grades *recieved*
		by agent i.

	Notes
	-----------
	"""
	# Normalize so that each Column sums to 1!
	col_sums = score_matrix.sum(axis=0)
	
	if partition != {}:
		#Sanity check: iterate over the agents by partition and ensure
		#that they gave someone a score... otherwise give 1/(n-|P|-1)
		#to each agent outside.
		n = score_matrix.shape[0]
		#Iterate over partitions
		for c_key in partition.keys():
		 #For each agent in each partition...
		 for c_agent in partition[c_key]:
			 #Check that they gave scores...
			 if col_sums[c_agent] == 0:
				 #If they didn't, give everyone not themselves and not in their
				 #partition a point.
				 for c_other in range(n):
					 if c_other != c_agent and c_other not in partition[c_key]:
						 score_matrix[c_other][c_agent] = 1.
	else:
		#Give a 1 to everyone but themselves if their sum is 0.
		for j, total in enumerate(col_sums):
			if total == 0:
				score_matrix[:, j] = 1.
				score_matrix[j, j] = 0.

	#Resum and normalize..
	col_sums = score_matrix.sum(axis=0)
	norm_score_matrix = score_matrix / col_sums[np.newaxis , : ]
	# We may still have nan's because everyone's in one partition...
	norm_score_matrix = np.nan_to_num(norm_score_matrix)
	if _DEBUG: print("\nnormalized score matrix:\n" + str(norm_score_matrix))
	return norm_score_matrix

def topk(score_matrix, k, normalize=True):
	"""
	Implicit input:
		A function f: [0,1]^{n*n} --> Sigma_k

	Input:
		k: number of agents to select = number of agents to randomly choose

	Process:
		1) Take a set X = {x_1, ..., x_k}
		2) Set A = None (set of players in top k)
		3) Set B = None (set of players not in top k)
		4) For each i in [k], if others think x_i is in the top k, put in A. Else, put in B.
		5) While not enough people in top k:
			5a) If we need more people in A than we do in B, take best person in the opinion of X and add to A
			5b) Else, take worst person in the opinion of X and add to B
		6) Add that person to X to give you more information later.

	Notes:
		f = Borda (as in paper)
	"""

	if _DEBUG: print("\n\tRunning Top-k\n")
	## Set N for convenience...
	n = score_matrix.shape[0]
	# normalize without partitions
	norm_score_matrix = normalize_score_matrix(score_matrix)

	# choose k people
	k_ids = random.sample(list(range(n)), k)

	k_norm_score_matrix = norm_score_matrix[:, k_ids]

	A = []
	B = []

	# get their evaluations for each other using X_{-i}
	for i, k_id in enumerate(k_ids):
		neg_k_scores = np.subtract(np.sum(k_norm_score_matrix, axis=1), norm_score_matrix[:, k_id])
		k_sorted_indices = [i[0] for i in sorted(enumerate(neg_k_scores), key=lambda x:x[1], reverse=True)]
		# print(k_id, k_sorted_indices, k_sorted_indices.index(k_id))
		# tie-breaking is an issue here...?
		k_id_loc = k_sorted_indices.index(k_id)

		# put them in the right set
		if k_id_loc < k:
			A.append(k_id)
		else:
			B.append(k_id)

	while len(A) < k:
		# print('len', len(k_ids))
		# print('id', k_ids)
		# print('A',A)
		# print('B',B)
		k_norm_score_matrix = norm_score_matrix[:, k_ids]
		if k - len(A) > n - k - len(B):
			# argmin
			# print('branch one', sorted(enumerate(np.sum(k_norm_score_matrix, axis=1)), key=lambda x:x[1]))
			argmin_list = [j[0] for j in sorted(enumerate(np.sum(k_norm_score_matrix, axis=1)), key=lambda x:x[1], reverse=True)]
			argmin_list = [a for a in argmin_list if a not in k_ids]
			i = argmin_list[0]
			# append to A
			A.append(i)
		else:
			# argmax
			# print('branch two', sorted(enumerate(np.sum(k_norm_score_matrix, axis=1)), key=lambda x:x[1], reverse=True))
			argmax_list = [j[0] for j in sorted(enumerate(np.sum(k_norm_score_matrix, axis=1)), key=lambda x:x[1])]
			argmax_list = [a for a in argmax_list if a not in k_ids]
			i = argmax_list[0]
			# append to B
			B.append(i)
		k_ids.append(i)

	return A


def committee_naive(score_matrix, k, normalize=True):  # previously bucket
	"""
	Implicit input:
		A function f: [0,1]^{n*n} --> Sigma_k

	Input:
		k: number of agents to select = number of agents to randomly choose

	Process:
		1) Take a set X = {x_1, ..., x_k}
		2) For each i in [k], let x_i = position k lfloor f(X_{-i}) / k rfloor + i
			f is Borda in this example
		3) Have all X's evaluate everyone else (Borda) and put in closest slot after...

	Notes:
		f = Borda (as in paper)
		Truncate after top k / don't care about what bucket people are in after the first k
	"""

	if _DEBUG: print("\n\tRunning Committee\n")
	## Set N for convenience...
	n = score_matrix.shape[0]
	# normalize without partitions
	norm_score_matrix = normalize_score_matrix(score_matrix)

	# choose k people
	k_ids = random.sample(list(range(n)), k)

	k_norm_score_matrix = norm_score_matrix[:, k_ids]

	final_ranking = [None] * n

	# get their evaluations for each other using X_{-i}
	for i, k_id in enumerate(k_ids):
		neg_k_scores = np.subtract(np.sum(k_norm_score_matrix, axis=1), norm_score_matrix[:, k_id])
		k_sorted_indices = [i[0] for i in sorted(enumerate(neg_k_scores), key=lambda x:x[1], reverse=True)]
		# tie-breaking is an issue here...?
		k_id_loc = k_sorted_indices.index(k_id)

		# put them in the closest bucket + offset
		if abs(k_id_loc - (k * math.floor(k_id_loc / k) + i)) <= abs((math.ceil(k_id_loc / k) + i) - k_id_loc):
			k_id_bucketed_loc = k * math.floor(k_id_loc / k) + i
		else:
			if k * math.ceil(k_id_loc / k) + i < n:
				k_id_bucketed_loc = k * math.ceil(k_id_loc / k) + i
			else:
				k_id_bucketed_loc = k * math.floor(k_id_loc / k) + i

		# put them in the correct buckets + offsets
		# k_id_bucketed_loc = k * math.floor(k_id_loc / k) + i

		final_ranking[k_id_bucketed_loc] = k_id

	# print('initial k', final_ranking[0:k])

	placed_so_far = k_ids
	not_yet_placed = list(set(range(n)) - set(placed_so_far))

	# get the people to evaluate for the next guy
	full_k_scores = np.sum(norm_score_matrix[:, k_ids], axis=1)

	while len(not_yet_placed) > 0:
		next_id = random.choice(not_yet_placed)

		sorted_indices = [i[0] for i in sorted(enumerate(full_k_scores), key=lambda x:x[1], reverse=True)]

		sorted_indices = [x for x in sorted_indices if x in not_yet_placed]

		insert_loc = sorted_indices.index(next_id) + 1

		# get the right insertion index
		insert_index = -1
		count = 0
		for index, i in enumerate(final_ranking):
			if i == None:
				count = count + 1
				if count == insert_loc:
					insert_index = index

		final_ranking[insert_index] = next_id

		not_yet_placed.remove(next_id)
		placed_so_far = np.append(placed_so_far, next_id)


	# select top k
	return final_ranking[0:k]

def committee_dynamic(score_matrix, k, normalize=True):  # previously bucket
	"""
	Implicit input:
		A function f: [0,1]^{n*n} --> Sigma_k

	Input:
		k: number of agents to select = number of agents to randomly choose

	Process:
		1) Take a set X = {x_1, ..., x_k}
		2) For each i in [k], let x_i = position k lfloor f(X_{-i}) / k rfloor + i
			f is Borda in this example
		3) Have all X's evaluate everyone else (Borda) and put in remaining slots (just order and put in open slots)

	Notes:
		f = Borda (as in paper)
		Truncate after top k / don't care about what bucket people are in after the first k
	"""

	if _DEBUG: print("\n\tRunning Committee\n")
	## Set N for convenience...
	n = score_matrix.shape[0]
	# normalize without partitions
	norm_score_matrix = normalize_score_matrix(score_matrix)

	# choose k people
	k_ids = random.sample(list(range(n)), k)

	k_norm_score_matrix = norm_score_matrix[:, k_ids]

	final_ranking = [None] * n

	# get their evaluations for each other using X_{-i}
	for i, k_id in enumerate(k_ids):
		neg_k_scores = np.subtract(np.sum(k_norm_score_matrix, axis=1), norm_score_matrix[:, k_id])
		k_sorted_indices = [i[0] for i in sorted(enumerate(neg_k_scores), key=lambda x:x[1], reverse=True)]
		# tie-breaking is an issue here...?
		k_id_loc = k_sorted_indices.index(k_id)

		# put them in the correct buckets + offsets
		k_id_bucketed_loc = k * math.floor(k_id_loc / k) + i

		final_ranking[k_id_bucketed_loc] = k_id

	# print('initial k', final_ranking[0:k])

	placed_so_far = k_ids
	not_yet_placed = list(set(range(n)) - set(placed_so_far))

	while len(not_yet_placed) > 0:
		next_id = random.choice(not_yet_placed)
		
		# get the people to evaluate for the next guy
		full_k_scores = np.sum(norm_score_matrix[:, placed_so_far], axis=1)

		sorted_indices = [i[0] for i in sorted(enumerate(full_k_scores), key=lambda x:x[1], reverse=True)]

		sorted_indices = [x for x in sorted_indices if x in not_yet_placed]

		insert_loc = sorted_indices.index(next_id) + 1

		# get the right insertion index
		insert_index = -1
		count = 0
		for index, i in enumerate(final_ranking):
			if i == None:
				count = count + 1
				if count == insert_loc:
					insert_index = index

		final_ranking[insert_index] = next_id

		not_yet_placed.remove(next_id)
		placed_so_far = np.append(placed_so_far, next_id)


	# select top k
	return final_ranking[0:k]

def bipartite(score_matrix, k, normalize=True):
	if _DEBUG: print("\n\tRunning Bipartite\n")
	## Set N for convenience...
	n = score_matrix.shape[0]
	# normalize without partitions
	norm_score_matrix = normalize_score_matrix(score_matrix)

	n1 = math.ceil(n/2)
	n2 = math.floor(n/2)

	# partition
	X = random.sample(list(range(n)), n1)
	Y = list(set(list(range(n))) - set(X))

	X_scores = np.sum(norm_score_matrix[:, X], axis=1)
	Y_scores = np.sum(norm_score_matrix[:, Y], axis=1)

	X_sorted_indices = [i[0] for i in sorted(enumerate(X_scores), key=lambda x:x[1], reverse=True)]

	Y_sorted_indices = [i[0] for i in sorted(enumerate(Y_scores), key=lambda x:x[1], reverse=True)]

	X_restricted_to_Y = [x for x in X_sorted_indices if x in Y]
	Y_restricted_to_X = [y for y in X_sorted_indices if y in X]

	sigma = []

	for i in range(n):
		if i % 2 == 0:
			sigma.append(Y_restricted_to_X[int(i/2)])
		if i % 2 == 1:
			sigma.append(X_restricted_to_Y[int((i-1)/2)])

	return sigma[0:k]


def kpartite(score_matrix, k, normalize=True):
	if _DEBUG: print("\n\tRunning K-partite\n")
	## Set N for convenience...
	n = score_matrix.shape[0]
	# normalize without partitions
	norm_score_matrix = normalize_score_matrix(score_matrix)
	
	# split into k groups: X_1, ..., X_k
	num_large_groups = n % k
	num_small_groups = k - n % k
	groups = []
	all_ids = list(range(n))

	Z_list = []

	for x in range(num_large_groups):
		sel_tmp = np.random.choice(all_ids, math.ceil(n/k),replace=False)
		groups.append(sel_tmp)
		all_ids = [a for a in all_ids if a not in sel_tmp]
	
	for x in range(num_small_groups):
		sel_tmp = np.random.choice(all_ids, math.floor(n/k),replace=False)
		groups.append(sel_tmp)
		all_ids = [a for a in all_ids if a not in sel_tmp]

	for i in range(k):
		X = groups[i]
		X_scores = np.sum(norm_score_matrix[:, X], axis=1)
		tau_i = [i[0] for i in sorted(enumerate(X_scores), key=lambda x:x[1], reverse=True)]
		gamma_i = n / len(X)

		Z_i = np.zeros((n,n))

		for a in range(n):
			for b in range(n):
				if a not in X:
					if tau_i.index(b) == a:
						Z_i[a,b] = 1 / gamma_i
					elif tau_i.index(b) in X:
						Z_i[a,b] = 1 / (gamma_i * (gamma_i - 1) * len(X))
					else:
						Z_i[a,b] = 0

		Z_list.append(Z_i)

		# print(X_scores)
		# print(tau_i)
		# print(Z_i)

	Z = np.zeros((n,n))

	for i, Z_i in enumerate(Z_list):
		const = ((n / len(groups[i])) - 1) / (k - 1)
		Zprime = const * Z_i
		Z = np.add(Z, Zprime)
		# print('const:', const)
		# print('Z_i:', Z_i)
		# print('Zprime:', Zprime)

	# print(Z)

	BvN_decomposition = birkhoff_von_neumann_decomposition(Z)

	# for coefficient, permutation_matrix in BvN_decomposition:
	# 	print('coefficient:', coefficient)
	# 	print('permutation matrix:', permutation_matrix)

	BvN_coeffs = [coeff for coeff, perm_matrix in BvN_decomposition]
	BvN_perms = [perm_matrix for coeff, perm_matrix in BvN_decomposition]

	# print(BvN_coeffs)
	# print(BvN_perms)

	winner_id = np.random.choice(len(BvN_coeffs), p=BvN_coeffs)
	# print(winner_id)

	winner_perm = BvN_decomposition[winner_id]

	# print(winner_perm)

	ordered_list = []

	for x in winner_perm[1]:
		ordered_list.append(list(x).index(1.))

	return(ordered_list[0:k])

def main():
	# a = np.array([[0, 3, 2, 1, 0, 0],
	#               [3, 0, 0, 2, 0, 1],
	#               [0, 3, 0, 0, 2, 1],
	#               [3, 0, 2, 0, 1, 0],
	#               [3, 0, 0, 2, 0, 1],
	#               [0, 3, 2, 0, 1, 0]])

	a = np.array([[6, 5, 4, 3, 2, 1],
								[6, 5, 4, 3, 2, 1],
								[6, 5, 4, 3, 2, 1],
								[6, 5, 4, 3, 2, 1],
								[6, 5, 4, 3, 2, 1],
								[6, 5, 4, 3, 2, 1]
								]).T
	k = 3

	print('topk', topk(a, k))
	print('committee', committee_naive(a,k))
	print('bipartite', bipartite(a,k))
	print('kpartite', kpartite(a,k))

if __name__ == '__main__':
	main()