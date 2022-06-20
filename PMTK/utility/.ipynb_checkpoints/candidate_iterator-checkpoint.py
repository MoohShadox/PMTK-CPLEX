import numpy as np
import random
from docplex.mp.model import Model
from docplex.util.environment import get_environment
from PMTK.pref.preferences import *
from PMTK.utility.utility_solver import *
from PMTK.utility.model_solver import *
from PMTK.utils import *
from itertools import chain, combinations

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




class CandidateIterator():

    def __init__(self,connivent, representant = None, theta = None):
        self.representant = representant
        self.theta = theta
        self.connivent = connivent
        subsets = [x for x,y in connivent] + [y for x,y in connivent]
        n_subsets = []
        for s in subsets:
            if not exist_superset(s, subsets):
                n_subsets.append(s)
        self.subsets = n_subsets
        random.shuffle(self.subsets)
        self.current_subset = 0
        self.current_powerset = 0

    def __iter__(self):
        return self

    def __next__(self):
        while self.current_powerset < len(self.subsets):
            s = self.subsets[self.current_powerset]
            p = list(powerset(s))
            if self.current_subset >= len(p):
                self.current_powerset += 1
                continue
            #print(" when powerset is:", self.current_powerset, " and subset is:", self.current_subset)
            #print("powerset of:", s, " subset:", p[self.current_subset])
            s = p[self.current_subset]
            self.current_subset += 1
            #if self.representant and theta_better(self.representant, self.theta+[s]) == 1:
            if self.representant and additivity(self.representant) < additivity(self.theta + [s]):
                self.current_powerset += 1
            if not s in self.theta and check_connivence_resolution(self.connivent,s):
                #print("We return ",s)
                return s
        raise StopIteration
