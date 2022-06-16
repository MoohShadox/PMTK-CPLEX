from itertools import chain, combinations
import numpy as np

def exist_superset(subset, subsets):
    for s in subsets:
        if all(i in s for i in subset) and not all(i in subset for i in s):
            return True
    return False

subset_utilities = lambda x,mdl:[mdl.utilities[s] for s in mdl.utilities if all(i in x for i in s)]

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1,len(s)+1))

def additivity(theta):
    return max([len(t) for t in theta])


def theta_better(theta_1, theta_2):
    if additivity(theta_1) != additivity(theta_2):
        return 1 if additivity(theta_1) < additivity(theta_2) else -1
    else:
        if len(theta_1) != len(theta_2):
            return 1 if len(theta_1) < len(theta_2) else -1
        else:
            return 0
        
def evaluate(v, utility, subsets):
    c = 0
    for u,s in zip(utility, subsets):
        if all(i in v for i in s):
            c += u
    return c

def compare_with_utilities(s_1, s_2, utilities, subsets):
    k = 0
    for u in utilities:
        vs_1 = evaluate(s_1, u, subsets)
        vs_2 = evaluate(s_2, u, subsets)
        if vs_1 > vs_2:
            k += 1
    return k

def utility_entropy(s_1, s_2, utilities, subsets):
    k1 = compare_with_utilities(s_1, s_2, utilities, subsets)
    k2 = compare_with_utilities(s_2, s_1, utilities, subsets)
    k = k1 + k2
    if k == 0:
        return 0
    p1 = k1 / k
    p2 = k2 / k
    if p1 == 0 or p2 == 0:
        return 0
    return -(p1*np.log(p1) + p2*np.log(p2)) 

def entropy_matrix(sampled, utilities, subsets):
    entropy_matrix = np.zeros((len(sampled), len(sampled)))
    for s_1 in sampled:
        for s_2 in sampled:
            if s_1 == s_2:
                continue
            entropy_matrix[sampled.index(s_1), sampled.index(s_2)] = utility_entropy(s_1, s_2, utilities, subsets)
    return entropy_matrix