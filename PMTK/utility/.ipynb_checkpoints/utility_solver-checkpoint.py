import numpy as np
import random
from docplex.mp.model import Model
from docplex.util.environment import get_environment
from PMTK.pref.preferences import *
from itertools import chain, combinations

subset_utilities = lambda x,mdl:[mdl.utilities[s] for s in mdl.utilities if all(i in x for i in s)]

def get_MPR(s_1, s_2,mdl):
    e = mdl.sum(subset_utilities(s_2,mdl)) - mdl.sum(subset_utilities(s_1,mdl))
    c = mdl.add_constraint(e <= 1)
    mdl.maximize_static_lex([-mdl.slack_sum, e])
    mdl.solve()
    v = e.solution_value
    mdl.remove_constraint(c)
    return v

def ordinal_dominance(s_1, s_2, mdl):
    e = get_MPR(s_1,s_2,mdl)
    e2 = get_MPR(s_2, s_1, mdl)
    #print(f"x={s_1}, y = {s_2}, e1 = {e}, e2 = {e2}")
    if e < 0:
        return "SUP"
    if e2 < 0:
        return "INF"
    if e <= 0 and e2 > 0:
        return "SUP_EQ"
    if e2 <= 0 and e > 0:
        return "INF_EQ"
    if e == e2 and e == 0:
        return "EQ"
    
    
def ordinal_peferences(items, subsets, mdl):
    prf = Preferences(items)
    for i_1 in range(len(subsets)):
        s_1 = subsets[i_1]
        for i_2 in range(i_1 + 1, len(subsets)):
            s_2 = subsets[i_2]
            r = ordinal_dominance(s_1, s_2, mdl)
            if r == "SUP":
                prf.add_preference(s_1, s_2)
            elif r == "INF":
                prf.add_preference(s_2, s_1)
            elif r == "EQ":
                prf.add_indifference(s_1,s_2)
    return prf

def get_subset_vars(subset, var_d):
    exp = 0
    for k in var_d:
        if all([i in subset for i in k]):
            exp += var_d[k]
    return exp

def get_preferences_vars(t, p_set, p_cst):
    exp = 0
    for p, p_var in zip(p_set, p_cst):
        x,y = p
        e = 0
        e += p_var if all(i in x for i in t) and len(x) > 0 else 0
        e -= p_var if all(i in y for i in t) and len(y) > 0 else 0
        exp += e
        #print(f"(x,y) = {x,y}, t = {t} , e = {e}")
    return exp

def subset_to_str(subset):
    return f"{subset}".replace("(","").replace(")", "").replace(",","").replace(" ","_")

def preference_complexity(subset_1, subset_2, theta):
    s_1 = set(subset_1)
    s_2 = set(subset_2)
    s_i = s_1.intersection(s_2)
    n = (2**len(s_1) - 1) + (2**len(s_2) - 1) - (2*(2**len(s_i) - 1))
    for t in theta:
        if all([i in s_1 and i in s_2 for i in t]):
            continue
        if all([i in s_1 for i in t]):
            n = n - 1
        if all([i in s_2 for i in t]):
            n = n - 1
    return n

def get_connivent(theta, preferences):
    if EMPTY_SET in theta:
        theta.remove(EMPTY_SET)
    #print("p= ", preferences)
    #print("theta= ", theta)
    mdl = unsat_polyhedron(preferences.items, theta, preferences)
    comp_sum = 0
    for p, p_var in zip(mdl.prefs, mdl.prf_var):
        comp_sum += preference_complexity(p[0], p[1], theta) * p_var
    #print("complexity total:" , comp_sum)
    mdl.maximize_static_lex([mdl.connivence_test, -comp_sum])
    mdl.solve()
    #print("It gave the solution: ", mdl.connivence_test.solution_value)
    if mdl.connivence_test.solution_value == 0:
        return None
    connivent = []
    #print(preferences)
    #print(mdl.lp_string)
    #print(mdl.connivence_test.solution_value)
    #print(mdl.s_connivent_l.solution_value)
    for p, p_var in zip(mdl.prefs, mdl.prf_var):
        if p_var.solution_value != 0:
            connivent.append(p)
    return connivent

def unsat_polyhedron(items, theta, preferences, **kwargs):
    if EMPTY_SET in theta:
        theta.remove(EMPTY_SET)

    mdl = Model(**kwargs)
    prefs = [(x,y) for x,y in preferences.preferred] + [(x,y) for x,y in preferences.indifferent] + [(y,x) for x,y in preferences.indifferent]
    prf_var = mdl.continuous_var_list(prefs,lb = 0, ub = 1, name= lambda x:f"p_{x[0]}_{x[1]}")
    for s in theta:
        e = get_preferences_vars(s, prefs, prf_var)
        #print(f"for s = {s} Added cst:", e == 0)
        mdl.add_constraint(e == 0, ctname = f"c_{s}")

    strict_cst_var = mdl.continuous_var(lb = 0, ub = 1)
    mdl.prefs = prefs
    mdl.prf_var = prf_var

    mdl.s_connivent_l = mdl.sum_vars_all_different(prf_var)
    mdl.s_connivent_s = mdl.sum_vars_all_different(prf_var[:len(preferences.preferred)])
    mdl.add_constraint(mdl.s_connivent_s + strict_cst_var >= 1)
    mdl.connivence_test = 1 - strict_cst_var

    mdl.add_kpi(mdl.s_connivent_l, 'Connivence score')
    mdl.add_kpi(1-strict_cst_var, 'Connivence Test')
    return mdl

def utility_polyhedron(items, theta, preferences, tolerance = 1e-5, lambda_bound = np.inf, **kwargs):
    mdl = Model(**kwargs)
    mvar = mdl.continuous_var_dict(theta, -np.inf, np.inf, name=lambda x:f"u_{subset_to_str(x)}")
    slack = []
    for i in range(len(preferences.subsets)):
        for j in range(i+1, len(preferences.subsets)):
            s_i = preferences.subsets[i]
            s_j = preferences.subsets[j]
            s_i_v = get_subset_vars(s_i, mvar)
            s_j_v = get_subset_vars(s_j, mvar)

            if preferences.is_preferred(s_i, s_j) == 1:
                v = mdl.continuous_var(lb = 0, ub = lambda_bound, name = f"l#{subset_to_str(s_i)}#{subset_to_str(s_j)}")
                mdl.add_constraint(s_i_v - s_j_v >= 1 - v)
                slack.append(v)

            if preferences.is_preferred(s_j, s_i) == 1:
                v = mdl.continuous_var(lb = 0, ub = lambda_bound, name = f"l#{subset_to_str(s_i)}#{subset_to_str(s_j)}")
                v = mdl.continuous_var(lb = 0, ub = lambda_bound, name = f"l#{subset_to_str(s_j)}#{subset_to_str(s_i)}")
                mdl.add_constraint(s_j_v - s_i_v >= 1 - v)
                slack.append(v)

            if preferences.is_indifferent(s_i, s_j) == 1:
                v = mdl.continuous_var(lb = 0, ub = lambda_bound, name = f"e1#{subset_to_str(s_i)}#{subset_to_str(s_j)}")
                v2 = mdl.continuous_var(lb = 0, ub = lambda_bound, name = f"e2#{subset_to_str(s_i)}#{subset_to_str(s_j)}")
                mdl.add_constraint(s_i_v + v == s_j_v + v2)
                slack.append(v)
                slack.append(v2)
    mdl.theta = theta
    mdl.preferences = preferences
    mdl.utilities = mvar
    mdl.slack = slack
    mdl.slack_sum = mdl.sum_vars_all_different(slack)
    mdl.u_sum = mdl.sum_vars_all_different(mvar.values())
    mdl.add_kpi(mdl.slack_sum, 'Slack Sum')
    mdl.add_kpi(mdl.u_sum, 'Model Complexity')
    return mdl

def get_kernel(preferences, theta, found = []):
    mdl = utility_polyhedron(preferences.items, theta, preferences)
    mdl.indicators = {}
    for s in theta:
        b = mdl.binary_var(name = f"b_{s}")
        mdl.add_constraint(mdl.abs(mdl.utilities[s]) <= 1e3*b)
        mdl.indicators[s] = b

    additivity_exps = [mdl.indicators[s]*len(s) for s in mdl.indicators]
    additivity_obj = mdl.max(additivity_exps)

    size_obj = mdl.sum_vars_all_different(mdl.indicators.values())
    valid = mdl.binary_var(name = "valid")
    #print("==== ")
    #print("Computing kernel", )
    #print(preferences)
    #print(theta)
    
    if len(found) > 0:
        o1 = mdl.binary_var(name = "o1")
        o2 = mdl.binary_var(name = "o2")
        representant = found[0]
        mdl.add_indicator(o1, additivity_obj == additivity(representant))
        mdl.add_indicator(o2, size_obj == len(representant))
        indics = [o1, o2]
        for f in found:
            f_b = [x for x in theta if not x in f]
            v_f = [mdl.indicators[i] for i in f ]
            v_fb = [mdl.indicators[i] for i in mdl.utilities if i not in f]
            c1 = mdl.logical_or(*v_fb)
            c2 = mdl.logical_and(*v_f)
            c3 = mdl.logical_not(c2)
            #print("c1 = ", c1, " c3 = ", c3)
            if len(v_fb) > 0:
                c = mdl.logical_or(c1, c3)
            else:
                c = c3
            indics.append(c)
        mdl.add_constraint(valid == mdl.logical_and(*indics))

    mdl.minimize_static_lex([mdl.slack_sum, -valid, additivity_obj, size_obj])
    #print(mdl.lp_string)
    mdl.solve(log_output = False)
    #print("Slack: ", mdl.slack_sum.solution_value)
    #print("Valid:", valid.solution_value)
    #print("Additivity:", additivity_obj.solution_value)
    #print("Size:", size_obj.solution_value)
    if mdl.slack_sum.solution_value != 0:
        raise Exception("Empty Polyhedron")
    if valid.solution_value != 1:
        return None
    kernel = [s for s in mdl.utilities if mdl.indicators[s].solution_value != 0]
    #print( [mdl.indicators[s].solution_value for s in mdl.utilities])
    return kernel

def get_kernels(preferences, theta):
    found = []
    k = get_kernel(preferences, theta, found = found)
    while k:
        found.append(k)
        k = get_kernel(preferences, theta, found)
        #print("Found = ", found)
    return found




def get_kernel_lex2(preferences, theta, found = []):
    mdl = utility_polyhedron(preferences.items, theta, preferences)
    mdl.indicators = {}
    for s in theta:
        b = mdl.binary_var(name = f"b_{s}")
        mdl.add_constraint(mdl.abs(mdl.utilities[s]) <= 1e3*b)
        mdl.indicators[s] = b

    additivity_exps = [mdl.indicators[s]*len(s) for s in mdl.indicators]
    additivity_obj = mdl.max(additivity_exps)

    size_obj = mdl.sum_vars_all_different(mdl.indicators.values())
    valid = mdl.binary_var(name = "valid")
    #print("==== ")
    #print("Computing kernel", )
    #print(preferences)
    #print(theta)
    
    if len(found) > 0:
        o1 = mdl.binary_var(name = "o1")
        o2 = mdl.binary_var(name = "o2")
        representant = found[0]
        mdl.add_indicator(o1, additivity_obj == additivity(representant))
        mdl.add_indicator(o2, size_obj == len(representant))
        indics = [o1, o2]
        for f in found:
            f_b = [x for x in theta if not x in f]
            v_f = [mdl.indicators[i] for i in f ]
            v_fb = [mdl.indicators[i] for i in mdl.utilities if i not in f]
            c1 = mdl.logical_or(*v_fb)
            c2 = mdl.logical_and(*v_f)
            c3 = mdl.logical_not(c2)
            #print("c1 = ", c1, " c3 = ", c3)
            if len(v_fb) > 0:
                c = mdl.logical_or(c1, c3)
            else:
                c = c3
            indics.append(c)
        mdl.add_constraint(valid == mdl.logical_and(*indics))

    mdl.minimize_static_lex([mdl.slack_sum, -valid, additivity_obj, size_obj])
    #print(mdl.lp_string)
    mdl.solve(log_output = False)
    #print("Slack: ", mdl.slack_sum.solution_value)
    #print("Valid:", valid.solution_value)
    #print("Additivity:", additivity_obj.solution_value)
    #print("Size:", size_obj.solution_value)
    if mdl.slack_sum.solution_value != 0:
        raise Exception("Empty Polyhedron")
    if valid.solution_value != 1:
        return None
    kernel = [s for s in mdl.utilities if abs(mdl.indicators[s].solution_value - 1)<= 1e-6]
    #print( [mdl.indicators[s].solution_value for s in mdl.utilities])
    return kernel

def get_kernel_lex3(preferences, theta, found = []):
    mdl = utility_polyhedron(preferences.items, theta, preferences)
    mdl.indicators = {}
    for s in theta:
        b = mdl.binary_var(name = f"b_{s}")
        mdl.add_constraint(mdl.abs(mdl.utilities[s]) <= 1e3*b)
        mdl.indicators[s] = b

    additivity_exps = [mdl.indicators[s]*len(s) for s in mdl.indicators]
    additivity_obj = mdl.max(additivity_exps)
    expressivity_obj = mdl.sum(additivity_exps)
    
    size_obj = mdl.sum_vars_all_different(mdl.indicators.values())
    valid = mdl.binary_var(name = "valid")
    #print("==== ")
    #print("Computing kernel", )
    #print(preferences)
    #print(theta)
    
    if len(found) > 0:
        o1 = mdl.binary_var(name = "o1")
        o2 = mdl.binary_var(name = "o2")
        o3 = mdl.binary_var(name = "o3")
        representant = found[0]
        mdl.add_indicator(o1, additivity_obj == additivity(representant))
        mdl.add_indicator(o2, size_obj == len(representant))
        mdl.add_indicator(o3, expressivity_obj == sum([len(i) for i in representant]))
        indics = [o1, o2, o3]
        for f in found:
            f_b = [x for x in theta if not x in f]
            v_f = [mdl.indicators[i] for i in f ]
            v_fb = [mdl.indicators[i] for i in mdl.utilities if i not in f]
            c1 = mdl.logical_or(*v_fb)
            c2 = mdl.logical_and(*v_f)
            c3 = mdl.logical_not(c2)
            #print("c1 = ", c1, " c3 = ", c3)
            if len(v_fb) > 0:
                c = mdl.logical_or(c1, c3)
            else:
                c = c3
            indics.append(c)
        mdl.add_constraint(valid == mdl.logical_and(*indics))

    mdl.minimize_static_lex([mdl.slack_sum, -valid, additivity_obj, size_obj, expressivity_obj])
    #print(mdl.lp_string)
    mdl.solve(log_output = False)
    #print("Slack: ", mdl.slack_sum.solution_value)
    #print("Valid:", valid.solution_value)
    #print("Additivity:", additivity_obj.solution_value)
    #print("Size:", size_obj.solution_value)
    if mdl.slack_sum.solution_value != 0:
        raise Exception("Empty Polyhedron")
    if valid.solution_value != 1:
        return None
    kernel = [s for s in mdl.utilities if abs(mdl.indicators[s].solution_value - 1)<= 1e-6]
    #print( [mdl.indicators[s].solution_value for s in mdl.utilities])
    return kernel


def get_kernel_add(preferences, theta, found = []):
    mdl = utility_polyhedron(preferences.items, theta, preferences)
    mdl.indicators = {}
    for s in theta:
        b = mdl.binary_var(name = f"b_{s}")
        mdl.add_constraint(mdl.abs(mdl.utilities[s]) <= 1e3*b)
        mdl.indicators[s] = b 
        
    additivity_exps = [mdl.indicators[s]*len(s) for s in mdl.indicators]
    additivity_obj = mdl.max(additivity_exps)
    
    valid = mdl.binary_var(name = "valid")
    #print("==== ")
    #print("Computing kernel", )
    #print(preferences)
    #print(theta)
    
    if len(found) > 0:
        o1 = mdl.binary_var(name = "o1")
        representant = found[0]
        mdl.add_indicator(o1, additivity_obj == additivity(representant))
        indics = [o1]
        for f in found:
            f_b = [x for x in theta if not x in f]
            v_f = [mdl.indicators[i] for i in f ]
            v_fb = [mdl.indicators[i] for i in mdl.utilities if i not in f]
            c1 = mdl.logical_or(*v_fb)
            c2 = mdl.logical_and(*v_f)
            c3 = mdl.logical_not(c2)
            #print("c1 = ", c1, " c3 = ", c3)
            if len(v_fb) > 0:
                c = mdl.logical_or(c1, c3)
            else:
                c = c3
            indics.append(c)
        mdl.add_constraint(valid == mdl.logical_and(*indics))

    mdl.minimize_static_lex([mdl.slack_sum, -valid, additivity_obj])
    #print(mdl.lp_string)
    mdl.solve(log_output = False)
    #print("Slack: ", mdl.slack_sum.solution_value)
    #print("Valid:", valid.solution_value)
    #print("Additivity:", additivity_obj.solution_value)
    #print("Size:", size_obj.solution_value)
    if mdl.slack_sum.solution_value != 0:
        raise Exception("Empty Polyhedron")
    if valid.solution_value != 1:
        return None
    kernel = [s for s in mdl.utilities if abs(mdl.indicators[s].solution_value - 1)<= 1e-6]
    #print( [mdl.indicators[s].solution_value for s in mdl.utilities])
    return kernel


def get_kernel_variance(preferences, theta, found = []):
    mdl = utility_polyhedron(preferences.items, theta, preferences)
    mdl.indicators = {}
    for s in theta:
        b = mdl.binary_var(name = f"b_{s}")
        mdl.add_constraint(mdl.abs(mdl.utilities[s]) <= 1e3*b)
        mdl.indicators[s] = b

    variance_obj = sum([mdl.indicators[s]*(2**len(s)) for s in mdl.indicators])
    valid = mdl.binary_var(name = "valid")
    #print("==== ")
    #print("Computing kernel", )
    #print(preferences)
    #print(theta)
    #print("==============In variance===========")
    
    
    if len(found) > 0:
        o1 = mdl.binary_var(name = "o1")
        representant = found[0]
        #print(f"Constraining with {representant} objective to be {sum([2**len(i) for i in representant])}")
        mdl.add_indicator(o1, variance_obj == sum([2**len(i) for i in representant]))
        indics = [o1]
        for f in found:
            f_b = [x for x in theta if not x in f]
            v_f = [mdl.indicators[i] for i in f ]
            v_fb = [mdl.indicators[i] for i in mdl.utilities if i not in f]
            c1 = mdl.logical_or(*v_fb)
            c2 = mdl.logical_and(*v_f)
            c3 = mdl.logical_not(c2)
            #print("c1 = ", c1, " c3 = ", c3)
            if len(v_fb) > 0:
                c = mdl.logical_or(c1, c3)
            else:
                c = c3
            indics.append(c)
        mdl.add_constraint(valid == mdl.logical_and(*indics))

    mdl.minimize_static_lex([mdl.slack_sum, -valid, variance_obj])
    #print(mdl.lp_string)
    mdl.solve(log_output = False)
    #print("Slack: ", mdl.slack_sum.solution_value)
    #print("Valid:", valid.solution_value)
    #print("Additivity:", additivity_obj.solution_value)
    #print("Size:", size_obj.solution_value)
    if mdl.slack_sum.solution_value != 0:
        raise Exception("Empty Polyhedron")
        
    if valid.solution_value != 1:
        return None
    kernel = [s for s in mdl.utilities if abs(mdl.indicators[s].solution_value - 1)<= 1e-6]
    #print( [mdl.indicators[s].solution_value for s in mdl.utilities])
    #print("Objective was: ", variance_obj.solution_value)
    #print("Returning ", kernel , "with objective value", sum([2**len(i) for i in kernel]))
    return kernel



def get_kernels_lex2(preferences, theta):
    k = True
    found = []
    while k:
        k = get_kernel_lex2(preferences, theta, found)
        if k:
            found.append(k)
            #print(f"Found n°{len(found)} = ", found[-1], "with obj:", sum(2**(len(i)) for i in found[-1]))
    return found

def get_kernels_lex3(preferences, theta):
    k = True
    found = []
    while k:
        k = get_kernel_lex3(preferences, theta, found)
        if k:
            found.append(k)
            #print(f"Found n°{len(found)} = ", found[-1], "with obj:", sum(2**(len(i)) for i in found[-1]))
    return found

#def get_kernels_add(preferences, theta):
#    found = []
#    k = get_kernel_add(preferences, theta, found = found)
#    while k:
#        found.append(k)
#        k = get_kernel_add(preferences, theta, found)
#        found_metrics = [sum(2**(len(i)) for i in j) for j in found]
#        #print(f"Found n°{len(found)} = ", found[-1], "with obj:", sum(2**(len(i)) for i in found[-1]))
#        print("Found metrics:", found_metrics)
#    return found

def get_kernels_var(preferences, theta):
    k = True
    found = []
    while k:
        k = get_kernel_variance(preferences, theta, found)
        if k:
            found.append(k)
            found_metrics = [sum(2**(len(i)) for i in j) for j in found]
            #print(f"Found n°{len(found)} = ", found[-1], "with obj:", sum(2**(len(i)) for i in found[-1]))
            if not all(i == found_metrics[-1] for i in found_metrics):
                print("Found metrics:", found_metrics)
                input("press")
    return found



def build_approx_theta(prf, init_theta = None):
    connivents = []
    if not init_theta:
        init_theta = [EMPTY_SET]
    theta = init_theta
    min_k = 1
    c  = get_connivent(theta, prf)
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
                for t in i:
                    b = False or check_connivence_resolution(c, t)
                    if not t in theta and check_connivence_resolution(c, t):
                        theta.append(t)
        c  = get_connivent(theta, prf)
    a = additivity(theta)
    for c in connivents:
        cit = get_candidate_iterator(c)
        for k in cit:
            if k > a:
                break
            for i in cit[k]:
                for t in i:
                    if not t in theta and check_connivence_resolution(c,t):
                        theta.append(t)
    
    return theta