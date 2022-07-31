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

def check_connivence_resolution(connivent_set, candidate):
    if len(candidate) == 0:
        return False
    c = 0
    for x,y in connivent_set:
        if all(i in x for i in candidate):
            c += 1
        if all(i in y for i in candidate):
            c -= 1
    return c != 0

        
def theta_better_opt(theta_1, theta_2):
    if additivity(theta_1) != additivity(theta_2):
        return 1 if additivity(theta_1) < additivity(theta_2) else -1
    else:
        if len(theta_1) != len(theta_2):
            return 1 if len(theta_1) < len(theta_2) else -1
        else:
            if sum([len(i) for i in theta_1]) != sum([len(i) for i in theta_2]):
                return 1 if sum([len(i) for i in theta_1]) < sum([len(i) for i in theta_2]) else -1
            return 0
        
        
def additivity(theta):
    return max([len(i) for i in theta])

def dfs_thetas_r(preferences, theta, theta_mins = [], stats = [], banned = [], bann_opt = True):
    if len(stats) == 0:
        stats.append(0)
    else:
        stats[0] = stats[0] + 1
    c = get_connivent(theta, preferences)
    print(" === Call with theta= ", theta , "===")
    representant = theta_mins[0] if len(theta_mins) > 0 else None
    if c == None:
        #print("Connivent found while theta_min is ", theta_mins)
        kernels = [theta]
        k2 = []
        #k2 = get_kernels(preferences, theta)
        if len(k2) > 0:
            kernels = k2

        if len(theta_mins) == 0:
            for k in kernels:
                theta_mins.append(k)
            return

        for t in kernels:
            theta_mins.append(t)
            t2 = keep_non_dominated(theta_mins)
            theta_mins.clear()
            for t in t2:
                theta_mins.append(t)
            print(theta , "✓")
            print("Upating theta min", theta_mins)
            print("== Closing node theta = ", theta, " == ")
        return

    cit = CandidateIterator(c, theta = theta, representant = representant)
    #print("Iterating over candidate")
    for candidate in cit:
        #print("Theta:", theta, "Connivent is ", c)
        #print("Trying candidate:", candidate)
        if candidate in banned and bann_opt:
            continue
        n_theta = theta + [candidate]
        representant = theta_mins[0] if len(theta_mins) > 0 else None
        cit.representant = representant
        if representant and theta_better(n_theta, representant) < 0:
            continue
        print("In theta= ", theta,"Connivent is ", c," Trying ", candidate)
        dfs_thetas_r(preferences, n_theta, theta_mins, stats = stats, banned = banned + [candidate])
    #print("== Closing node theta = ", theta, " == ")

def get_min_thetas(preferences, initial_theta, bann_opt = True):
    representant = None
    theta_mins = []
    stats = []
    dfs_thetas_r(preferences, initial_theta, theta_mins = theta_mins, stats = stats, bann_opt = bann_opt)
    return theta_mins, stats


def get_candidate_iterator(c, bound = np.inf):
    a = list(set([x[0] for x in c if len(x[0]) > 0] + [x[1] for x in c if len(x[1]) > 0]))
    a = sorted(a, key = lambda x:len(x))
    iterators = {}
    for x in a:
        for k in range(1,min(len(x)+1, bound)):
            iterators[k] = iterators.get(k,[]) + [itertools.combinations(x, k)]
    for k in iterators:
        random.shuffle(iterators[k])
    return iterators


def dfs_thetas_full(preferences, theta, theta_mins, banned = None, bann_opt = False):
    if not banned:
        banned = []
    theta = sorted(theta)
    c = get_connivent(theta, preferences)
    #print("Theta = ", theta, end = "\t")
    #print("c = ", c, end = "\t")
    #print("Banned = ", banned)
    
    if c == None:
        #print("Found ", theta)
        theta_mins.append(theta)
        return 
    
    its = get_candidate_iterator(c)
    for k in its:
        for candidates in its[k]:
            for c_i in candidates:
                if c in theta:
                    continue
                n_theta = theta + [c_i]
                dfs_thetas_full(preferences, n_theta, theta_mins)
    return True 

def dfs_thetas_r(preferences, theta, theta_mins = [], banned = None, bann_opt = True):
    if not banned:
        banned = []
    theta = sorted(theta)
    c = get_connivent(theta, preferences)
    #print("Theta = ", theta, end = "\t")
    #print("c = ", c, end = "\t")
    #print("Banned = ", banned)
    
    if c == None:
        kernels = [theta]
        #kernels += get_kernels(prf, theta)
        for theta in kernels:
            if theta in theta_mins:
                continue
            if len(theta_mins) == 0:
                theta_mins.append(theta)
                return additivity(theta_mins[0]), len(theta_mins[0])
            if theta_better(theta, theta_mins[0]) == 1:
                theta_mins.clear()
                theta_mins.append(theta)
            elif theta_better(theta, theta_mins[0]) == 0:
                theta_mins.append(theta)
        return 
                
    its = get_candidate_iterator(c)
    b = list(banned)
    for k in its:
        for candidates in its[k]:
            for c_i in candidates:
                if len(theta_mins) > 0 and k > additivity(theta_mins[0]):
                    #print("Cutting over", k, " in:", theta , end = " ")
                    break
                if c_i in theta or c_i in b:
                    continue
                n_theta = theta + [c_i]
                if len(theta_mins) > 0 and theta_better(n_theta, theta_mins[0]) == -1:
                    #print("Cutting", c , " in:", theta, end = " ")
                    continue
                #print("Trying candidate ", c)
                dfs_thetas_r(preferences, n_theta, theta_mins, banned = b)
                if bann_opt:
                    b.append(c_i)
    return True

def dfs_thetas_opt(preferences, theta, theta_mins = [], banned = None, bann_opt = True):
    if not banned:
        banned = []
    theta = sorted(theta)
    c = get_connivent(theta, preferences)
    #print("Theta = ", theta, end = "\t")
    #print("c = ", c, end = "\t")
    #print("Banned = ", banned)
    
    if c == None:
        kernels = [theta]
        #kernels += get_kernels(prf, theta)
        for theta in kernels:
            if theta in theta_mins:
                continue
            if len(theta_mins) == 0:
                theta_mins.append(theta)
                return additivity(theta_mins[0]), len(theta_mins[0])
            if theta_better_opt(theta, theta_mins[0]) == 1:
                theta_mins.clear()
                theta_mins.append(theta)
            elif theta_better_opt(theta, theta_mins[0]) == 0:
                theta_mins.append(theta)
        return
    
    its = get_candidate_iterator(c)
    b = list(banned)
    for k in its:
        for candidates in its[k]:
            for c_i in candidates:
                if len(theta_mins) > 0 and k > additivity(theta_mins[0]):
                    #print("Cutting over", k, " in:", theta , end = " ")
                    break
                if c_i in theta or c_i in b:
                    continue
                n_theta = theta + [c_i]
                if len(theta_mins) > 0 and theta_better_opt(n_theta, theta_mins[0]) == -1:
                    #print("Cutting", c , " in:", theta, end = " ")
                    continue
                #print("Trying candidate ", c)
                dfs_thetas_opt(preferences, n_theta, theta_mins, banned = b)
                if bann_opt:
                    b.append(c_i)
    return True



def get_min_thetas(preferences, initial_theta = None, bann_opt = True):
    representant = None
    theta_mins = []
    stats = []
    if not initial_theta:
        initial_theta = [EMPTY_SET]
    dfs_thetas_r(preferences, initial_theta, theta_mins = theta_mins, bann_opt = bann_opt)
    return theta_mins, stats

def get_all_thetas(preferences, initial_theta = None, bann_opt = True):
    representant = None
    theta_mins = []
    stats = []
    if not initial_theta:
        initial_theta = [EMPTY_SET]
    dfs_thetas_full(preferences, initial_theta, theta_mins = theta_mins, bann_opt = bann_opt)
    return theta_mins, stats

def get_opt_thetas(preferences, initial_theta = None, bann_opt = True):
    representant = None
    theta_mins = []
    stats = []
    if not initial_theta:
        initial_theta = [EMPTY_SET]
    dfs_thetas_opt(preferences, initial_theta, theta_mins = theta_mins, bann_opt = bann_opt)
    return theta_mins, stats



