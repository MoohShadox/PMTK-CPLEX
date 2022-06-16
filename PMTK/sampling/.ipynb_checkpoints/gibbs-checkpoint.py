from docplex.mp.model import Model
import numpy as np
import matplotlib.pyplot as plt
from PMTK.sampling.preferences_sampler import *
from PMTK.pref.preferences import *
from PMTK.utility.utility_solver import *
from PMTK.sampling.subset_samplers import *
from PMTK.utility.model_solver import *

def get_random_ext_points(mdl, mdl_vars, n_tryouts = 100):
    var_names = list(mdl_vars.keys())
    var_v = list(mdl_vars.values())
    points = []
    for _ in range(n_tryouts):
        coeffs = np.random.normal(0,100,len(var_names))
        mdl.maximize(sum(coeffs * var_v))
        mdl.solve()
        pt = {n:v.solution_value for v,n in zip(var_v, var_names)}
        if not pt in points:
            points.append(pt)
    return points

def sample_from_ext_pt(ext_pts):
    vals = np.array([np.array(list(e.values())) for e in ext_pts])
    coeffs = np.random.rand(vals.shape[0])
    coeffs = coeffs / coeffs.sum()
    return (vals.T @ coeffs)
    
def get_random_direction(mdl_vars):
    var_v = list(mdl_vars.values())
    coeffs = np.random.normal(0,1,len(var_v))
    coeffs = coeffs / coeffs.sum()
    return coeffs

def get_r_min_max(mdl, mdl_vars, p_i, direction):
    added_constraints = []
    r = mdl.continuous_var(lb = - mdl.infinity, ub = mdl.infinity )
    for p,v,d in zip(p_i, mdl_vars.values(), direction):
        c = mdl.add_constraint(v == p + d*r)
        added_constraints.append(c)
    mdl.maximize(r)
    mdl.solve()
    r_max = r.solution_value
    mdl.minimize(r)
    mdl.solve()
    r_min = r.solution_value
    mdl.remove(added_constraints)
    return (r_min, r_max)

def step_hit_and_run(mdl, mdl_vars, p_i):
    t = get_random_direction(mdl_vars)
    r_min, r_max = get_r_min_max(mdl, mdl_vars, p_i, t)
    r = np.random.uniform(r_min,r_max)
    return p_i + t * r
    
def get_center(ext_pts):
    vals = np.array([np.array(list(e.values())) for e in ext_pts])
    coeffs = np.ones(vals.shape[0])
    coeffs = coeffs / coeffs.sum()
    return (vals.T @ coeffs)

def sample_innter_points(mdl, mdl_vars, n_tryouts = 100, n_points = 20):
    ep = get_random_ext_points(mdl, mdl_vars, n_tryouts = n_tryouts)
    c = get_center(ep)
    points = []
    for _ in range(n_points):
        points.append(c)
        c = step_hit_and_run(mdl, mdl_vars, c)
    return points

def sample_utilities(theta, prf, n_points = 10):
    mdl = utility_polyhedron(prf.items, theta, prf)
    c = mdl.add_constraints([v<= 1 for v in mdl.utilities.values()])
    c += mdl.add_constraints([v>= -1 for v in mdl.utilities.values()])
    points = sample_innter_points(mdl, mdl.utilities, n_points=n_points)
    return np.array(points), mdl.utilities.keys()