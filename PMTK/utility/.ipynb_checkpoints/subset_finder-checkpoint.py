from docplex.mp.model import Model
from docplex.util.environment import get_environment
from PMTK.sampling.preferences_sampler import *
from PMTK.pref.preferences import *
from PMTK.utility.utility_solver import *
from PMTK.sampling.subset_samplers import *
from PMTK.utility.model_solver import *
from tqdm import tqdm
import numpy as np
import pandas as pd
from PMTK.utility.candidate_iterator import *
from PMTK.sampling.gibbs import *

def find_best_subset(items, utility, subsets, banned = []):
    mdl = Model("Subset Optim")
    item_v = mdl.binary_var_dict(items)
    u_var = mdl.binary_var_dict(subsets)
    for u in u_var:
        vs = [item_v[s] for s in item_v if s in u]
        a_v = mdl.logical_and(*vs)
        mdl.add_constraint(u_var[u] == a_v)
    for b in banned:
        x_v = [item_v[s] for s in item_v if s in b]
        x_nv = [item_v[s] for s in item_v if not s in b]
        if len(x_nv) > 0:
            if len(x_nv) == 0:
                a_x = mdl.logical_and(*x_v)
                mdl.add_constraint(a_x == 0)
                continue
            if len(x_v) == 0:
                b_x = mdl.logical_or(*x_nv)
                mdl.add_constraint(b_x == 1)
                continue
            a_x = mdl.logical_and(*x_v)
            na_x = mdl.logical_not(a_x)
            b_x = mdl.logical_or(*x_nv)
            o = mdl.logical_or(na_x, b_x)
            mdl.add_constraint(o == 1)

    obj = mdl.sum([u*u_var[s] for s,u in zip(subsets, utility)])
    mdl.maximize(obj)
    mdl.solve()
    return tuple([s for s in item_v if item_v[s].solution_value == 1])

def get_opt_deg(items, utility, subsets, deg = 1, banned = []):
    b = []
    found = []
    for _ in range(deg):
        f = find_best_subset(items, utility, subsets, banned = b + banned)
        found.append(f)
        b.append(f)
    return found

def rank_by_deg(items, utilities, subsets, deg, banned = []):
    all_d = set()
    vecs = []
    for u in utilities:
        d = get_opt_deg(items, u, subsets, deg, banned= banned)
        all_d.update(d)
        vecs.append(d)
    sorted_subsets = {}
    for d in all_d:
        k = min([d_v.index(d) if d in d_v else np.inf for d_v in vecs])
        sorted_subsets[d] = k
    return {k: v for k, v in sorted(sorted_subsets.items(), key=lambda item: item[1])}