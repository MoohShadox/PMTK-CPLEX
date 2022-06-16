"""
@Author: Ouaguenouni Mohamed
"""
from PMTK.preferences import Preferences
import numpy as np
import pandas as pd
import cvxpy as cp


class AdditiveUtility:
    """
    This class aims at defining and using an n-additive function.
    The definition of an Additive_Utility consists in:
        1- Defining the set theta that contains all the subset for which w
        authorize a non null utility.
        2- Defining the utility values of each subset in theta.
    The evaluation of a subset consists in summing the utility values of each
    subset in theta contained in it.
    """

    def __init__(self, items):
        self.theta = []
        self.alternatives = set(items)
        self.theta_values = {}

    def add_to_theta(self, *new_t):
        """
        This function add a sequences of sets to theta.
        args:
            new_t: a sequence of subsets each one represented by a tuple.
        """
        for subset in new_t:
            for i in subset:
                if i not in self.alternatives:
                    return False
        for subset in new_t:
            s_t = tuple(sorted(set(subset)))
            if s_t not in self.theta:
                self.theta.append(s_t)

        self.theta = sorted(self.theta)
        return True

    def set_theta_values(self, *vals):
        """
        Affect the utilities to one or many subsets in theta.
        args:
            vals: a sequence of key,value where the key represents a subset
            in theta and the value the utility associated with it.
        """
        for key, val in vals:
            n_v = tuple(sorted(key))
            if n_v in self.theta:
                self.theta_values[n_v] = val
            else:
                self.theta.append(tuple(sorted(key)))
                self.theta_values[n_v] = val

    def __call__(self, x):
        return self.evaluate(x)

    def __str__(self):
        ch = ""
        ch += "Model: " + str(self.theta) + "\n"
        for k in self.theta_values:
            ch += f"{k}: {self.theta_values[k]} \n"
        return ch

    def get_dominant_subset(self):
        pass

    def evaluate(self, x_s):
        """
        Evaluates the utility of a set of alternatives x by summing the values
        of the subsets in theta that are contained in x_s.
        Exemple:
            theta = [(1, ) , (2, ), (1, 3)];
            theta_values={(1,):2 , (2,): 5, (1,3):1}
            we have:
                evaluate([1,3]) = 2 + 1 = 3
                evaluate([2]) = 5
                evaluate([1,2,3]) = 5 + 2 + 1 = 8
        """
        if len(x_s) == 0:
            return 0
        cpt = 0
        for i in self.theta:
            if all(j in x_s for j in i):
                if i in self.theta_values:
                    cpt += self.theta_values[i]
        return cpt

    def compute_relation(self, subsets, add_empty = True, epsilon = 1e-8):
        """
        Given a subset of elements return the set of all the existant
        comparison relations.
        """
        pref = Preferences(self.alternatives)
        if add_empty:
            subsets += [tuple([])]
        for s_i in subsets:
            for s_j in subsets:
                if s_i != s_j:
                    e_1 = self.evaluate(s_i)
                    e_2 = self.evaluate(s_j)
                    #print(f"for {s_i}: {e_1} and for {s_j}: {e_2}")
                    if e_1 >= e_2+epsilon:
                        pref.add_preference(s_i, s_j)
                    elif (abs(e_1 - e_2) <= epsilon) :
                        pref.add_indifference(s_i, s_j)
                    else:
                        pref.add_preference(s_j, s_i)

        return pref
