import pandas as pd
import numpy as np
from collections import Counter
from PMTK.sampling.preferences_sampler import *
from PMTK.pref.preferences import *
from PMTK.utility.utility_solver import *
from PMTK.sampling.subset_samplers import *
from PMTK.utility.model_solver import *
from PMTK.utility.subset_finder import *
from PMTK.sampling.gibbs import *


class Configuration_Problem_User: 
    
    def __init__(self, components, additivity = 1):
        self.components = components
        self.costs = np.random.randint(1, 100, size = (len(self.components), ))
        self.utilities = {i:np.abs(np.random.normal(0,10)*10) for i in get_all_k_sets(self.components,additivity)}
    
    def __str__(self):
        ch = "Model: \n"
        ch += f"Costs: {self.costs} \n"
        for u in self.utilities:
            ch += f"{u} : {self.utilities[u]} \n"
        return ch
    
    def __repr__(self):
        return self.__str__()
    
    def __call__(self, x):
        if len(x) == 0:
            return np.array([-np.inf, -np.inf])
        u_s = 0
        for u in self.utilities:
            if all([i in x for i in u]):
                u_s += self.utilities[u]
        cost = 0
        for i in x:
            cost += self.costs[list(self.components).index(i)]
            
        return np.array([u_s, -cost])


class Tierlist_Decider: 
    
    def __init__(self, items, p = 0.2, sigma = 100, alpha = 0.2, n_theta = None, n_tiers = 5):
        self.items = items
        self.utilities = {}
        self.p = p
        self.n_theta = n_theta
        self.n_tiers = n_tiers
        if not self.n_theta:
            self.n_theta = int(alpha*(len(items) + len(items)**2))+1
        for k in get_all_k_sets(items, 1):
            self.utilities[k] = np.random.normal(0, sigma)
        for i in range(self.n_theta):
            theta = self.sample_geometric_subset()
            while theta in self.utilities:
                theta = self.sample_geometric_subset()
            self.utilities[theta] = np.random.normal(0, sigma)
        self.tiers = self.compute_tiers()
        
    def evaluate(self, s):
        cpt = 0
        for k in self.utilities:
            if all(i in s for i in k):
                cpt += self.utilities[k]
        return cpt


    def sample_geometric_subset(self):
        it = list(self.items)
        s = random.choice(it)
        it.remove(s)
        theta = [s]
        while len(it) != 0:
            s = random.choice(it)
            theta.append(s)
            it.remove(s)
            r = random.random()
            if r > self.p:
                break
        return tuple(sorted(theta))

    def find_max_min(self):
        subsets = list(self.utilities.keys())
        utility = list(self.utilities.values())
        min_v = self.evaluate(find_best_subset(self.items, utility, subsets, sense = -1))
        max_v = self.evaluate(find_best_subset(self.items, utility, subsets, sense = 1))
        return max_v, min_v
    
    def compute_tiers(self):
        max_v, min_v = self.find_max_min()
        return np.linspace(min_v, max_v, self.n_tiers + 1)
    
    def __str__(self):
        ch = "Model: \n"
        for u in self.utilities:
            ch += f"{u} : {self.utilities[u]} \n"
        return ch
    
    def __repr__(self):
        return self.__str__()
    
    def __call__(self, x):
        if len(x) == 0:
            return 0
        u_s = 0
        for u in self.utilities:
            if all([i in x for i in u]):
                u_s += self.utilities[u]
        return np.argmax((self.tiers > u_s).astype(int))


    
class MO_Objective_Function:
    def __init__(self, items, f):
        self.f = f
        self.items = items
        self.budget = 0
        self.saved = {}
        self.epsilon = 1e-6
        
    def relation(self):
        preferences = Preferences(self.items)
        for i in self.saved:
            for j in self.saved:
                if i == j:
                    continue
                if pareto_dominate(self.saved[i], self.saved[j], self.epsilon) > 0:
                    preferences.add_preference(i, j)
                elif pareto_dominate(self.saved[i], self.saved[j], self.epsilon) < 0:
                    preferences.add_preference(j,i) 
        return preferences
    
    def pareto_front(self):
        costs = - np.array(list(self.saved.values()))
        elements = np.array(list(self.saved.keys()))
        return elements[is_pareto_efficient_simple(costs)]
    
    def evaluated_elements(self): 
        return len(self.saved.keys())
    
    def __call__(self, x):
        if not x in self.saved:
            self.saved[x] = self.f(x)
            self.budget += 1
        return self.saved[x]
    
    
class Objective_Function:
    def __init__(self, items, f):
        self.f = f
        self.items = items
        self.budget = 0
        self.saved = {}
        self.epsilon = 2
        
    def relation(self):
        preferences = Preferences(self.items)
        for i in self.saved:
            for j in self.saved:
                if i == j:
                    continue
                if self.saved[i] > self.saved[j]:
                    preferences.add_preference(i,j)
                elif self.saved[j] > self.saved[i]:
                    preferences.add_preference(j, i)
        return preferences
    
    
    def evaluated_elements(self): 
        return len(self.saved.keys())
    
    def __call__(self, x):
        if not x in self.saved:
            self.saved[x] = self.f(x)
            self.budget += 1
        return self.saved[x]