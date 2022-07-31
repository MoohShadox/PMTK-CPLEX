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
from PMTK.data.film_dataset import *
from PMTK.sampling.decider import *


def build_approx_theta(prf, init_theta = None):
    connivents = []
    if not init_theta:
        init_theta = [EMPTY_SET]
    theta = init_theta
    min_k = 1
    c  = get_connivent(theta, prf)
    cpt = 0
    while c:
        if not c in connivents:
            connivents.append(c)
        cit = get_candidate_iterator(c)
        skey = sorted(cit.keys())[0]
        b = False
        for k in cit:
            if b:
                break
            for i in cit[k]:
                s = set(i)
                for t in s:
                    b = b or (check_connivence_resolution(c, t) and not t in theta)
                    if not t in theta and check_connivence_resolution(c, t):
                        theta.append(t)
        c  = get_connivent(theta, prf)
        cpt = cpt + 1
        #print("solved connivent: ", cpt, " with", theta)
    a = additivity(theta)
    for c_i in connivents:
        cit = get_candidate_iterator(c_i)
        for k in cit:
            if k > a:
                break
            for i in cit[k]:
                for t in i:
                    if not t in theta and check_connivence_resolution(c_i,t):
                        theta.append(t)
    
    return theta