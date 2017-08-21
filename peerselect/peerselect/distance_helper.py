import itertools
import random
import re
import sys
import os

def dKT(a, b):
	# same length, same set of elts, no repeats
	assert len(a) == len(b)
	assert set(a) == set(b)
	assert len(set(a)) == len(a)

	pairs = itertools.combinations(range(0, len(a)), 2)

	distance = 0

	for x, y in pairs:
		asign = a.index(x) - a.index(y)
		bsign = b.index(x) - b.index(y)

		# if discordant (different signs)
		if (asign * bsign < 0):
			distance += 1

	return distance

def displ(a, b):
	# same length, same set of elts, no repeats
	assert len(a) == len(b)
	assert set(a) == set(b)
	assert len(set(a)) == len(a)

	d = []
	for i, x in enumerate(a):
		d.append(abs(i - b.index(x)))
	return d

def dFR(a, b):
	d = displ(a, b)
	return sum(d)
	
def dMD(a, b):
	d = displ(a, b)
	return max(d)

def dCY(a, b):
	# same length, same set of elts, no repeats
	assert len(a) == len(b)
	assert set(a) == set(b)
	assert len(set(a)) == len(a)

	# count cycles
	base_len = len(a)
	num_cycles = 0
	counter_list = [entry for entry in a]

	b = [a.index(x) for x in b]
	base = [1,2,3,4]

	all_cycles = []
	for (i,x) in enumerate(b):
		seen = []
		next = None
		prev = None
		while next not in seen:
			seen.append(x)
			next = b[x]
			x = next
		all_cycles.append(seen)
	
	all_cycles = [sorted(c) for c in all_cycles]
	distCY = list(set(frozenset(item) for item in all_cycles))

	return base_len - len(distCY)


def dHM(a, b):
	# same length, same set of elts, no repeats
	assert len(a) == len(b)
	assert set(a) == set(b)
	assert len(set(a)) == len(a)

	vec = []
	for i, x in enumerate(a):
		if b[i] != x:
			vec.append(1)  # 1 if different
		else:
			vec.append(0)  # 0 if same

	return sum(vec)
