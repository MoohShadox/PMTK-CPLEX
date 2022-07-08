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