import gurobipy as gb
import numpy as np
import math
import sys
from birkhoff import birkhoff_von_neumann_decomposition
import csv
import random
sys.path.insert(0, '/Users/akahng/Dropbox/RESEARCH/Procaccia/17-impartiality/peerselection-master/peerselect')
from peerselect import distance_helper as dist

'''
Converts comparisons to usable form.
In:
cmps: (number of comparisons) \times 4 matrix with rows representing reviews, the first
      column representing the reviewer's id, the second representing a user id, the third 
      representing another user id, and the fourth being 1 if the review reviewed the 
      first user higher, and -1 otherwise.
valid: list of all users you care about. If valid = None, then all users are counted.

Out:
mapping: An array where the ith entry tells us the ith user id.
mtx: n \times n matrix where the i, j entry represents the number of times i was said to
     be better than j.
'''
def convert(cmps,valid=None):
    num_reviews = len(cmps)
    num_users = 0
    # create reverse_mapping (maps ids to indices)
    reverse_mapping = {}
    for r in range(num_reviews):
        for c in range(0,3):
            user = cmps[r][c]
            if not (user in reverse_mapping):
                reverse_mapping[user] = num_users
                num_users += 1
#     inverse_mapping = dict((v, k) for k, v in reverse_mapping.items())
    # create mapping
    mapping = [0 for x in range(num_users)]
    for key, value in reverse_mapping.items():
        mapping[value] = key
    if valid is None:
        valid = mapping
    # create mtx
    mtx = [[0 for x in range(num_users)] for y in range(num_users)]
    for r in range(num_reviews):
#         if inverse_mapping[cmps[r][0]] in valid:  # make sure you get the cmps you want
        if cmps[r][0] in valid:  # make sure you get the cmps you want
            a = reverse_mapping[cmps[r][1]]
            b = reverse_mapping[cmps[r][2]]
            if cmps[r][3] == 1:
                mtx[a][b] += 1
            elif cmps[r][3] == -1:
                mtx[b][a] += 1
    # return
    return (mapping, mtx)

'''
Applies Borda to a cmps matrix.
In:
mapping: An array where the ith entry is the ith user id.
mtx: n \times n matrix where the i, j entry represents the number of times i was said to
     be better than j.
Out:
ranking: Borda ranking (ids)
'''
def borda(mapping, mtx):
    borda_scores = [sum(row) for row in mtx]
    sorted_borda = [b[0] for b in sorted(enumerate(borda_scores), key=lambda i:i[1], reverse=True)]
    return [mapping[i] for i in sorted_borda]
    
'''
Computes the Kemeny ranking.
In:
n: number of people.
mtx: n \times n matrix where the i, j entry represents the number of times i was said to
     be better than j.
Out:
ranking: Kemeny ranking
'''
def kemeny(mapping, mtx):
    n = len(mapping)
    try:
        model = gb.Model("kemeny")
        # create variables
        xs_names = [[str(x) + "," + str(y) for x in range(n)] for y in range(n)]
        xs = [[0 for x in range(n)] for y in range(n)]
        for i in range(n):
            for j in range(n):
                if (i != j):
                    xs[i][j] = model.addVar(vtype = gb.GRB.BINARY, name = xs_names[i][j])
        # set objective
        obj = gb.LinExpr()
        for i in range(n):
            for j in range(n):
                if (i != j and mtx[i][j] > 0):
                    obj += mtx[i][j]*xs[i][j]
        model.setObjective(obj, gb.GRB.MAXIMIZE)
        # add exclusion constraints
        for i in range(n):
            for j in range(n):
                if (i != j):
                    model.addConstr(xs[i][j] + xs[j][i] == 1)
        # add transitivity constraints
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if (i != j and j != k and i != k):
                        model.addConstr(xs[i][j] + xs[j][k] - xs[i][k] <= 1)
        # solve
        model.setParam('OutputFlag', False)
        model.optimize()
        variables = model.getVars()
        ranking = [-1 for i in range(n)]
        for i in range(n):
            cnt = 0
            for j in range(n):
                if (i != j and xs[i][j].X == 0):
                    cnt += 1
            ranking[cnt] = i
        ranking = [mapping[i] for i in ranking]
        return ranking
    except gb.GurobiError:
        print('Error reported')

'''
Outputs results.
In:
ranking: Kemeny ranking
mapping: An array where the ith entry tells us the ith user id.
'''
def output_results(ranking, mapping):
    n = len(ranking)
    print('The ranking:', ranking)
    # for i in range(n):
    #     print(ranking[i])
        # print(str(mapping[ranking[i]]))
        
'''
Solves and outputs results.
In:
cmps: (number of comparisons) \times 3 matrix with rows representing reviews, the first
      column representing one user id, the second representing another user id, and the
      third being 1 if the review reviewed the first user higher, and -1 otherwise.
'''
def solve_kemeny(cmps):
    (mapping, mtx) = convert(cmps)
    ranking = kemeny(mapping, mtx)
    ranking = [mapping.index(i) for i in ranking]
    return ranking
    # output_results(ranking, mapping)

def solve_borda(cmps):
    (mapping, mtx) = convert(cmps)
    ranking = borda(mapping, mtx)
    return ranking
    
def solve_kemeny_k(cmps,k):
    (mapping, mtx) = convert(cmps)
    ranking = kemeny(mapping, mtx)
    ranking = ranking[0:k]
    return ranking

def topk(cmps, k, ranking_fun):
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
        f = Borda or Kemeny (input)
    """
    
    (mapping, mtx) = convert(cmps)
    n = len(mtx)

    # choose k people
    k_ids = random.sample(mapping, k)
    # print(k_ids)
    
    A = []
    B = []

    # get their evaluations for each other using X_{-i}
    for i, k_id in enumerate(k_ids):
        effective_k_ids = list(set(k_ids) - set([k_id]))
        (effective_mapping, effective_score_matrix) = convert(cmps, effective_k_ids)
        k_rankings = ranking_fun(effective_mapping, effective_score_matrix)
        k_id_loc = k_rankings.index(k_id)

        # put them in the right set
        if k_id_loc < k:
            A.append(k_id)
        else:
            B.append(k_id)
            
    # print(A,B)

    while len(A) < k:
        (k_mapping, k_norm_score_matrix) = convert(cmps, k_ids)
        if k - len(A) > n - k - len(B):
            # argmin
            argmin_list = ranking_fun(k_mapping, k_norm_score_matrix)
            argmin_list = [a for a in argmin_list if a not in k_ids]
            i = argmin_list[0]
            # append to A
            A.append(i)
        else:
            # argmax
            argmax_list = ranking_fun(k_mapping, k_norm_score_matrix)
            argmax_list = [a for a in argmax_list if a not in k_ids]
            i = argmax_list[-1]
            # append to B
            B.append(i)
        k_ids.append(i)

    A_ids = [mapping.index(x) for x in A]

    return A_ids

def committee_naive(cmps, k, ranking_fun):  # previously bucket; fix overflow issue!! needs k | n
    """
    Implicit input:
        A function f: [0,1]^{n*n} --> Sigma_k

    Input:
        k: number of agents to select = number of agents to randomly choose

    Process:
        1) Take a set X = {x_1, ..., x_k}
        2) For each i in [k], let x_i = position k lfloor f(X_{-i}) / k rfloor + i
            f is Borda or Kemeny in this example
        3) Have all X's evaluate everyone else and put in closest slot after...

    Notes:
        f = Borda or Kemeny (have to provide input)
        Truncate after top k / don't care about what bucket people are in after the first k
    """

    (mapping, mtx) = convert(cmps)
    n = len(mtx)

    # choose k people
    k_ids = random.sample(mapping, k)

    (k_mapping, k_norm_score_matrix) = convert(cmps, k_ids)

    # final_ranking = [None] * n
    final_ranking = [None] * k * math.ceil(n / k)

    # get their evaluations for each other using X_{-i}
    for i, k_id in enumerate(k_ids):
        effective_k_ids = list(set(k_ids) - set([k_id]))
        (effective_mapping, effective_score_matrix) = convert(cmps, effective_k_ids)
        k_rankings = ranking_fun(effective_mapping, effective_score_matrix)
        k_id_loc = k_rankings.index(k_id)

        # put them in the closest bucket + offset
        if abs(k_id_loc - (k * math.floor(k_id_loc / k) + i)) <= abs((math.ceil(k_id_loc / k) + i) - k_id_loc):
            k_id_bucketed_loc = k * math.floor(k_id_loc / k) + i
        else:
            if k * math.ceil(k_id_loc / k) + i < n:
                k_id_bucketed_loc = k * math.ceil(k_id_loc / k) + i
            else:
                k_id_bucketed_loc = k * math.floor(k_id_loc / k) + i

        # put them in the correct buckets + offsets
        final_ranking[k_id_bucketed_loc] = k_id

    # print('initial k', final_ranking[0:k])

    placed_so_far = k_ids
    not_yet_placed = list(set(mapping) - set(placed_so_far))

    # get the people to evaluate for the next guy
    (full_mapping, full_matrix) = convert(cmps, k_ids)

    while len(not_yet_placed) > 0:
        next_id = random.choice(not_yet_placed)

        sorted_indices = ranking_fun(full_mapping, full_matrix)

        sorted_indices = [x for x in sorted_indices if x in not_yet_placed]

        insert_loc = sorted_indices.index(next_id) + 1

        # get the right insertion index
        insert_index = -1
        count = 0
        for index, i in enumerate(final_ranking):
            if i is None:
                count = count + 1
                if count == insert_loc:
                    insert_index = index

        final_ranking[insert_index] = next_id

        not_yet_placed.remove(next_id)
        placed_so_far = np.append(placed_so_far, next_id)

    final_ranking = [x for x in final_ranking if x is not None]

    final_ranking_ids = [mapping.index(x) for x in final_ranking]

    # select top k
    # return final_ranking[0:k]
    return final_ranking_ids
    # return final_ranking_ids[0:k]

def bipartite(cmps, k, ranking_fun):
    """
    Implicit input:
        A function f: [0,1]^{n*n} --> Sigma_k

    Input:
        k: number of agents to select = number of agents to randomly choose

    Process:
        1) Split into two sets
        2) Have each set evaluate the other; then interleave deterministically

    Notes:
        f = Borda or Kemeny (have to provide input)
    """

    (mapping, mtx) = convert(cmps)
    n = len(mtx)

    n1 = math.ceil(n/2)
    n2 = math.floor(n/2)

    # partition
    X = random.sample(mapping, n1)
    Y = list(set(mapping) - set(X))

    (X_mapping, X_matrix) = convert(cmps, X)
    (Y_mapping, Y_matrix) = convert(cmps, Y)
    
    X_sorted_indices = ranking_fun(X_mapping, X_matrix)
    Y_sorted_indices = ranking_fun(Y_mapping, Y_matrix)

    X_restricted_to_Y = [x for x in X_sorted_indices if x in Y]
    Y_restricted_to_X = [y for y in X_sorted_indices if y in X]

    sigma = []

    for i in range(n):
        if i % 2 == 0:
            sigma.append(Y_restricted_to_X[int(i/2)])
        if i % 2 == 1:
            sigma.append(X_restricted_to_Y[int((i-1)/2)])

    sigma_ids = [mapping.index(x) for x in sigma]

    # return sigma[0:k]
    return sigma_ids
    # return sigma_ids[0:k]

def kpartite(cmps, kprime, ranking_fun):
    """
    Implicit input:
        A function f: [0,1]^{n*n} --> Sigma_k

    Input:
        kprime: number of agents to select = number of agents to randomly choose
        (note: have to deform to get valid input)

    Process:
        1) Split into k sets
        2) Create Z^(i) matrices based on the opinions of each partition
        3) Combine and sample using Birkhoff-von Neumann 

    Notes:
        f = Borda or Kemeny (have to provide input)
        Seems to do worse in practice than in theory...
    """
    (mapping, mtx) = convert(cmps)
    n = len(mtx)

    # split into k groups: X_1, ..., X_k of size ~ kprime
    k = math.ceil(n/kprime)

    num_large_groups = n % k
    num_small_groups = k - n % k

    groups = []
    all_ids = mapping

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
        (X_mapping, X_matrix) = convert(cmps, X)
        tau_i = ranking_fun(X_mapping, X_matrix)
        gamma_i = n / len(X)

        Z_i = np.zeros((n,n))

        for i,a in enumerate(mapping):
            for j,b in enumerate(mapping):
                X_indices = [mapping.index(x) for x in X]
                if a not in X:
                    if tau_i.index(b) == i:
                        Z_i[i,j] = 1 / gamma_i
                    elif tau_i.index(b) in X_indices:  # locations of X
                        Z_i[i,j] = 1 / (gamma_i * (gamma_i - 1) * len(X))
                    else:
                        Z_i[i,j] = 0

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
    # print(np.sum(Z, axis=0))
    # print(np.sum(Z, axis=1))

    BvN_decomposition = birkhoff_von_neumann_decomposition(Z)

    # for coefficient, permutation_matrix in BvN_decomposition:
    #   print('coefficient:', coefficient)
    #   print('permutation matrix:', permutation_matrix)

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
        ordered_list.append(mapping[list(x).index(1.)])

    ordered_list_ids = [mapping.index(x) for x in ordered_list]

    # return sigma[0:k]
    # return ordered_list[0:k]
    return ordered_list_ids
    # return ordered_list_ids[0:k]
    