import numpy as np
import random
from docplex.mp.model import Model
from docplex.util.environment import get_environment
from PMTK.pref.preferences import *
from PMTK.utility.utility_solver import *
from PMTK.utility.model_solver import *
from PMTK.utility.candidate_iterator import CandidateIterator
from PMTK.utils import *

def theta_better(theta_1, theta_2):
    if additivity(theta_1) != additivity(theta_2):
        return 1 if additivity(theta_1) < additivity(theta_2) else -1
    else:
        if len(theta_1) != len(theta_2):
            return 1 if len(theta_1) < len(theta_2) else -1
        else:
            return 0

def exist_superset(subset, subsets):
    for s in subsets:
        if all(i in s for i in subset) and not all(i in subset for i in s):
            return True
    return False

def next_candidate(connivent, theta, representant):
    s = 0
    subsets = [x for x,y in connivent] + [y for x,y in connivent]
    n_subsets = []
    #print("subsets: ", subsets)
    for s in subsets:
        if not exist_superset(s, subsets):
            n_subsets.append(s)
    subsets = n_subsets
    random.shuffle(subsets)
    #print("Subsets where to seek:", subsets)
    for s in subsets:
        p = powerset(s)
        for s in p:
            #print("Checking ", s)
            if not s in theta and check_connivence_resolution(connivent,s):
                #print("Return ",s)
                yield s
            if representant and theta_better(representant, theta+[s]):
                break

def dfs_thetas_r(preferences, theta, theta_mins = [], stats = [], banned = [], bann_opt = True):
    if len(stats) == 0:
        stats.append(0)
    else:
        stats[0] = stats[0] + 1
    c = get_connivent(theta, preferences)
    print(" === Call with theta= ", theta , "===")
    print("Theta:", theta)
    print("Theta min:", theta_mins)
    print("Connivent is", c)
    representant = theta_mins[0] if len(theta_mins) > 0 else None
    if c == None:
        print("Connivent found while theta_min is ", theta_mins)
        kernels = [theta]
        k2 = get_kernels(preferences, theta)

        if len(k2) > 0:
            kernels = k2

        if len(theta_mins) == 0:
            for k in kernels:
                theta_mins.append(k)
            return

        for t in kernels:
            if theta_better(t, theta_mins[0]) > 0:
                theta_mins.clear()
                theta_mins.append(t)

            if representant and theta_better(t, representant) == 0:
                theta_mins.append(t)
            print("Upating theta min", theta_mins)
            print("== Closing node theta = ", theta, " == ")
        return

    cit = CandidateIterator(c, theta = theta, representant = representant)
    #print("Iterating over candidate")
    for candidate in cit:
        #print("Trying candidate:", candidate)
        if candidate in banned and bann_opt:
            continue
        n_theta = theta + [candidate]
        representant = theta_mins[0] if len(theta_mins) > 0 else None
        cit.representant = representant
        if representant and theta_better(n_theta, representant) < 0:
            continue
        dfs_thetas_r(preferences, n_theta, theta_mins, stats = stats, banned = banned + [candidate])
    print("== Closing node theta = ", theta, " == ")

def get_min_thetas(preferences, initial_theta, bann_opt = True):
    representant = None
    theta_mins = []
    stats = []
    dfs_thetas_r(preferences, initial_theta, theta_mins = theta_mins, stats = stats, bann_opt = bann_opt)
    return theta_mins, stats


