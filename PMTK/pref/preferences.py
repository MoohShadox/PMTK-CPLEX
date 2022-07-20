""" @Author: Ouaguenouni Mohamed """ 
import numpy as np
import random
import itertools
import numpy as np
from enum import Enum

EMPTY_SET = tuple([])

def pareto_dominate(x,y, epsilon = 1e-2):
    if ((x - y) >= 0).all():
        return 1
    if ((y - x) >= 0).all():
        return -1
    return 0


def chain_iterators(*it):
    for i in it:
        for j in i:
            yield j

def additivity(theta):
    return max([len(i) for i in theta])

def better(theta1, theta2):
    if additivity(theta1) < additivity(theta2):
        return 2
    if additivity(theta2) < additivity(theta1):
        return -2
    if len(theta1) > len(theta2):
        return -1
    if len(theta2) > len(theta1):
        return 1
    return 0

def dominated(theta, theta_list):
    for t in theta_list:
        if better(t, theta) > 0:
            return True
    return False

def keep_non_dominated(theta_list):
    non_dominated = []
    for t in theta_list:
        if not dominated(t, theta_list):
            non_dominated.append(t)
    return non_dominated

def vectorize_subset( x, model):
    vector = np.zeros(len(model))
    for subset in model:
        if all(s in x for s in subset):
            vector[model.index(subset)] += 1
    return vector

def vectorize_preference(x, y, model):
    vector = vectorize_subset(x, model) - vectorize_subset(y, model)
    return vector

def is_pareto_efficient_simple(costs):
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]<c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient

def get_all_k_sets(items, k):
    """
    get all the subsets of size less or equal than k and that are included in the set of items.
    """
    subsets = []
    for i in range(1, k+1):
        k_subset = itertools.combinations(items, i)
        subsets = subsets + list(k_subset)
    return subsets

def get_exact_k_sets(items, k):
    """
    get all the subsets of size less or equal than k and that are included in the set of items.
    """
    k_subset = list(itertools.combinations(items, k))
    return k_subset

#TODO: Use subset instead of candidate
def get_k_candidates(connivents, k):
    L = []
    for connivent in connivents:
        s_1 = get_all_k_sets(connivent[0], k)
        s_2 = get_all_k_sets(connivent[1], k)
        L = L + s_1 + s_2
    return L

def get_exact_k_candidates(connivents, k):
    L = []
    for connivent in connivents:
        s_1 = get_exact_k_sets(connivent[0], k)
        s_2 = get_exact_k_sets(connivent[1], k)
        L = L + s_1 + s_2
    return L


def get_all_candidates(connivents):
    I = get_k_candidates(connivents, 1)
    return get_k_candidates(connivents, len(I))


def union(set_of_subsets):
    L = []
    for s in set_of_subsets:
        for i in s:
            if not i in L:
                L.append(i)
    return L


def preference_complexity(x,y):
    return 2**len(x) + 2**len(y) - (len(set(x).intersection(set(y))))


def vectorize_subset(x, m):
    vector = np.zeros(len(m))
    for subset in m:
        if all(s in x for s in subset):
            vector[m.index(subset)] += 1
    return vector

def vector_to_subset(vector, model):
    j = np.where(vector == 1)[0]
    s = []
    for k in j:
        for i in model[k]:
            s.append(i)
    return tuple(set(s))

class Relation(Enum):
    SUP = 2
    SUP_EQ = 1
    EQ = 0
    INF_EQ = -1
    INF = -2
    pass

class Preferences:
    """
    This class encapsulates different representations (sparse, matricial) of
    a set of preferences.
    These preferences could be represented with 2 predicates:
        - Strict preference.
        - Indifference.
    These two predicates could be extended by defining:
        - Prefered or indifferend as a disjonction of the two.
        - Incomparability if we don't have A preferred to B nor B
        preferred to A.
    """

    def __init__(self, items):
        self.items = items
        self.preferred = []
        self.preferred_or_indifferent = []
        self.indifferent = []
        self.subsets = []
        self.relation_matrix = None

    def to_dataset(self, model = None):
        if not model:
            model = [(i,) for i in self.items]
        arr = []
        for x,y in self.preferred:
            v_x = self.vectorize_subset(x, model)
            v_y = self.vectorize_subset(y, model)
            v1 = list(v_x) + list(v_y) + [1]
            v2 = list(v_y) + list(v_x) + [2]
            arr.append(np.array(v1))
            arr.append(np.array(v2))
        for x,y in self.indifferent:
            continue
            v_x = self.vectorize_subset(x, model)
            v_y = self.vectorize_subset(y, model)
            v1 = list(v_x) + list(v_y) + [0]
            v2 = list(v_y) + list(v_x) + [0]
            arr.append(np.array(v1))
            arr.append(np.array(v2))
        arr = np.array(arr)
        return arr
    
    def sample_subpref(self, ratio, min_number = 10):
        p = Preferences(self.items)
        cpt = 0
        while cpt < min_number:
            for x,y in self.preferred:
                if random.random() < ratio and cpt < min_number:
                    p.add_preference(x,y)
                    cpt = cpt + 1
            for x,y in self.indifferent:
                if random.random() < ratio and cpt < min_number:
                    p.add_indifference(x,y)
                    cpt = cpt + 1
            if cpt >= min_number:
                break
        return p
        
    def vectorize_subset(self, x, model):
        vector = np.zeros(len(model))
        for subset in model:
            if all(s in x for s in subset):
                vector[model.index(subset)] += 1
        return vector

    def vectorize_preference(self, x, y, model):
        vector = self.vectorize_subset(x, model) - self.vectorize_subset(y, model)
        return vector

    def get_matrix(self, model, relation_set):
        vectors = []
        for x,y in relation_set:
            v = self.vectorize_preference(x,y,model)
            vectors.append(v.astype(float))
        vectors = np.array(vectors).astype(float)
        return vectors

    def __getitem__(self, index):
        all_prefs = self.preferred + self.indifferent
        return all_prefs[index]

    def __set_item__(self, index, item):
        all_prefs = self.preferred + self.indifferent
        all_prefs[index] = item

    def contradictions(self, other):
        contr = []
        for x,y in self.preferred:
            if [y,x] in other.preferred:
                contr.append((x,y))
        #for x,y in self.indifferent:
        #    if not [x,y] in other.indifferent:
        #        contr.append((x,y))
        return contr
    
    def sort_by_n_candidates(self):
        self.preferred = sorted(self.preferred, key = lambda x:preference_complexity(x[0], x[1]))
        self.indifferent = sorted(self.indifferent, key = lambda x:preference_complexity(x[0], x[1]))
        return self


    def __add_subsets(self, subset):
        """
        Used to maintain a list of the subsets concerned by the preferences.
        """
        if subset == EMPTY_SET:
            if not subset in self.subsets:
                self.subsets.append(subset)
            return subset
        if type(subset) != tuple:
            subset = [subset]
        try:
            subset = [i for i in subset if i in self.items]
            subset = tuple(sorted(set(subset)))
            if subset not in self.subsets:
                self.subsets.append(subset)
            return subset
        except TypeException:
            print(f"Exception because of the type of {subset}")
            raise TypeException
        
    def add_preference(self, s_1, s_2):
        """
        Create a strict preference between x and y.
        """
        s_1 = self.__add_subsets(s_1)
        s_2 = self.__add_subsets(s_2)
        t_1 = [s_1, s_2]
        if t_1 not in self.preferred:
            self.preferred.append(t_1)

    def add_indifference(self, s_1, s_2):
        """
        Create an indifference relation between x and y.
        """
        s_1 = self.__add_subsets(s_1)
        s_2 = self.__add_subsets(s_2)
        t_1 = [s_1, s_2]
        if t_1 not in self.indifferent:
            self.indifferent.append(t_1)

    def is_preferred(self, s_1, s_2):
        """
        Test the preference between two subsets
        return
                1 if the first subset is preferred to the second,
                -1 if the opposite is true
                0 if neither case is true.
        """
        if [s_1, s_2] in self.preferred:
            return 1
        if [s_2, s_1] in self.preferred:
            return -1
        return 0

    def is_preferred_or_indifferent(self, s_1, s_2):
        """
        Test the preference or indifference between two subsets
        return
                1 if the first subset is preferred to the second,
                -1 if the opposite is true
                0 if neither case is true.
        """
        if [s_1, s_2] in self.preferred or [s_1, s_2] in self.indifferent:
            return 1
        if [s_2, s_1] in self.preferred or [s_2, s_1] in self.indifferent:
            return -1
        return 0

    def is_indifferent(self, s_1, s_2):
        """
        Test the indifference between two subsets
        return
                1:  If the decision-maker is indifferent to the choice
                    of one alternative over the other
                0:  If not.
        """
        if [s_1, s_2] in self.indifferent or [s_2, s_1] in self.indifferent:
            return 1
        return 0

    def is_incomparable(self, s_1, s_2):
        """
        Checks if we cannot compare two subsets.
        return
                1:  If we cannot compare s_1 to s_2 neither with the
                    indifference nor with the preference
                0:  If we can.
        """
        if [s_1, s_2] not in self.preferred \
           and [s_1, s_2] not in self.indifferent \
           and [s_2, s_1] not in self.indifferent \
           and [s_2, s_1] not in self.preferred:
            return 1
        return 0

    def __len__(self):
        return len(self.indifferent) + len(self.preferred)
        pass
    
    def __repr__(self):
        r_ch = ""
        for s_1, s_2 in self.preferred:
            r_ch += f"{s_1} > {s_2} \n"
        for s_1, s_2 in self.indifferent:
            r_ch += f"{s_1} = {s_2} \n"  
        return r_ch

    def __str__(self):
        r_ch = "Preference relation : \n"
        for s_1, s_2 in self.preferred:
            r_ch += f"{s_1} > {s_2} \n"
        for s_1, s_2 in self.indifferent:
            r_ch += f"{s_1} = {s_2} \n"
        return r_ch

    def __ge__(self, other):
        for s_1 in other.preferred:
            if s_1 not in self.preferred:
                return False
        for s_1 in other.indifferent:
            if s_1 not in self.indifferent:
                return False
        return True

    def __eq__(self, other):
        return other >= self and self >= other

    def __gt__(self, other):
        return self >= other and not self == other

    def __le__(self, other):
        return other >= self

    def __lt__(self, other):
        return other >= self and not other == self

    def __sub__(self, other):
        prf = Preferences(self.items)
        for s in self.preferred:
            if s not in other.preferred:
                prf.preferred.append(s)
        for s in self.indifferent:
            if s not in other.indifferent:
                prf.indifferent.append(s)
        return prf
    
    def intersect(self, other):
        P = Preferences(self.items)
        for x in self.preferred:
            if x in other.preferred:
                P.add_preference(*x)
        for x in self.indifferent:
            if x in other.indifferent:
                P.add_indifference(*x)
        return P
     
    def __add__(self, other):
        P = Preferences(self.items)
        for x,y in other.preferred:
            P.add_preference(x,y)
        for x,y in self.preferred:
            P.add_preference(x,y)  
        for x,y in other.indifferent:
            P.add_indifference(x,y)
        for x,y in self.indifferent:
            P.add_indifference(x,y)
        return P

class Weighted_Preferences(Preferences):
    
    def __init__(self, items):
        super().__init__(items)
        self.preferences_weight = []
        self.indifferences_weight = []

    def __add_subsets(self, subset):
        """
        Used to maintain a list of the subsets concerned by the preferences.
        """
        if subset == EMPTY_SET:
            if not subset in self.subsets:
                self.subsets.append(subset)
            return subset
        if type(subset) != tuple:
            subset = [subset]
        try:
            subset = [i for i in subset if i in self.items]
            subset = tuple(sorted(set(subset)))
            if subset not in self.subsets:
                self.subsets.append(subset)
            return subset
        except TypeException:
            print(f"Exception because of the type of {subset}")
            raise TypeException
            
    def add_preference(self, s_1, s_2, w):
        """
        Create a strict preference between x and y.
        """
        s_1 = self.__add_subsets(s_1)
        s_2 = self.__add_subsets(s_2)
        t_1 = [s_1, s_2]
        if t_1 not in self.preferred:
            self.preferred.append(t_1)
            self.preferences_weight.append(w)
            
    def add_indifference(self, s_1, s_2, w):
        """
        Create an indifference relation between x and y.
        """
        s_1 = self.__add_subsets(s_1)
        s_2 = self.__add_subsets(s_2)
        t_1 = [s_1, s_2]
        if t_1 not in self.indifferent:
            self.indifferent.append(t_1)
            self.indifferences_weight.append(w)
    
    def get_preferences_of_degree(self, d):
        prf = Preferences(self.items)
        for (x,y), w in zip(self.preferred, self.preferences_weight):
            if w >= d:
                prf.add_preference(x,y)
        for (x,y), w in zip(self.indifferent, self.indifferences_weight):
            if w >= d:
                prf.add_indifference(x,y)
        return prf
            
    def get_degree_of_preference(self, s_1, s_2):
        s_1 = self.__add_subsets(s_1)
        s_2 = self.__add_subsets(s_2)
        t_1 = [s_1, s_2]
        
        if t_1 in self.indifferent:
            return 0
        if t_1 in self.preferred:
            return w
        t_2 = [s_1, s_2]
        if t_2 in self.preferred:
            return -w
        

    

if __name__ == "__main__":
    pass
