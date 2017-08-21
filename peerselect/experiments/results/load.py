import pickle

base = pickle.load(open('long_run_versus_base.pickle', 'rb'))
vanilla = pickle.load(open('long_run_versus_vanilla.pickle', 'rb'))

# print(base)
# print(vanilla)

vanilla_v2 = {}

for x in vanilla:  
	# x is, e.g., (130, 25, 9, 5, 0.0, 100, 'DollarPartitionRaffle')
	tmp_list = [i/x[1] for i in vanilla[x]]
	tmp_avg = sum(tmp_list) / len(tmp_list)
	vanilla_v2[x] = tmp_avg

# print(vanilla_v2)

'''
s = 100
test_n = [130]
test_k = [15, 25, 35]
test_m = [5, 9, 13]
test_l = [3, 5]
test_p = [0.0, 0.2, 0.5]

VANILLA = "Vanilla"
EXACT = "ExactDollarPartition"
PARTITION = "Partition"
DPR = "DollarPartitionRaffle"
CREDIABLE = "CredibleSubset"
RAFFLE = "DollarRaffle"
ALL = (VANILLA, EXACT, PARTITION, RAFFLE, CREDIABLE, DPR)

'''

# top row in paper
test_m = 9
test_k = [35] #[15, 25, 35]
test_l = 3  # should be 4
test_p = [0.5]

for x in vanilla_v2:
	for k in test_k:
		# don't care about x[0] = 130 (n) and x[5] = 100 (s)
		if x[1] == k:  # k
			if x[2] == 9:  # m
				if x[3] == 3:  # l
					if x[4] == 0.5:  # p
						print(x, vanilla_v2[x])



