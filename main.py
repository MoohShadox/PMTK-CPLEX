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


def test_connivence():
    n_items = 10
    items = np.arange(n_items)
    for i in tqdm(range(200)):
        density = np.random.rand()
        n_theta = np.random.randint(0,2**n_items)
        theta = sample_subsets(items, n_subsets = n_theta)
        prf = sample_preferences_from_order(items, density*(2**(n_items+1)), indifference_rate= 0.3)
        mdl = utility_polyhedron(items, theta, prf)
        mdl.minimize(mdl.slack_sum)
        mdl.solve()
        connivence_mdl = unsat_polyhedron(items, theta, prf)
        connivence_mdl.maximize(connivence_mdl.connivence_test)
        connivence_mdl.solve()
        for prf, v in zip(connivence_mdl.prefs, connivence_mdl.prf_var):
            if v.solution_value > 0:
                continue
                print(v)
        if mdl.slack_sum.solution_value > 0 and not connivence_mdl.connivence_test.solution_value == 1:
                print("Wtf bro ?")

def test_ordinal_comparison():
    n_items = 5
    density = 0.8
    items = np.arange(n_items)
    theta = get_all_k_sets(items, 3)
    items = np.arange(n_items)
    prf = sample_preferences_from_order(items, density*(2**(n_items+1)), indifference_rate= 0.3)
    mdl = utility_polyhedron(items, theta, prf)
    subsets = sample_subsets(items, n_subsets = 20)
    for x in prf.subsets + subsets:
        for y in prf.subsets + subsets:
            if x==y:
                continue
            print(x,y,ordinal_dominance(x,y,mdl))
            print(y,x,ordinal_dominance(y,x,mdl))

def test_theta_min(preferences, theta):
    n_items = 6
    density = 0.7
    theta = [EMPTY_SET]
    items = np.arange(n_items)
    prf = sample_preferences_from_order(items, density*(2**(n_items+1)), indifference_rate= 0.2)
    theta_mins, stats = get_min_thetas(prf, theta)
    theta_mins_2, stats_2 = get_min_thetas(prf, theta, bann_opt = False)
    print(theta_mins)
    print(theta_mins_2)
    print(stats_2)
    print(stats)

def get_example_prf_set():
    prf = Preferences(items)
    prf.add_preference((0,) , EMPTY_SET)
    prf.add_preference((1,) , EMPTY_SET)
    prf.add_preference((2,) , EMPTY_SET)
    prf.add_preference((3,) , EMPTY_SET)
    prf.add_preference(EMPTY_SET, (0,1,2,3))
    return prf

def test_thetas_mins():
    n_items = 4
    #density = 0.8
    items = np.arange(n_items)
    theta = get_all_k_sets(items, n_items)
    prf = get_example_prf_set()
    t = get_min_thetas(prf, [EMPTY_SET])
    print(t)


if __name__ == "__main__":
    mdl = Model("Test")
    x = mdl.continuous_var(name = "x")
    y = mdl.continuous_var(name = "y")
    mdl.add_constraint(y + x == 5)
    mdl.add_constraint(x >= 0)
    mdl.add_constraint(x <= 3)
    mdl.add_constraint(y >= 0)

    X = np.arange(0, 10)
    y = np.arange(0, 10)
    plt.plot(X, -X + 5)
    plt.plot(3, y)
    plt.show()





