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

class Film_Dataset:
    
    def __init__(self, n_films, n_users,target_user, ratings_path = "data/ratings.csv"):
        self.n_films = n_films
        self.n_users = n_users
        
        df = pd.read_csv(ratings_path)
        df = df.groupby("userId").count().reset_index()
        count_user = {u:m for u,m in zip(df["userId"], df["movieId"])}
        
        df = pd.read_csv(ratings_path)
        df = df.groupby("movieId").count().reset_index()
        
        count_films = {u:m for u,m in zip(df["userId"], df["movieId"])}
        
        count_user = {i:j for i,j in sorted(count_user.items(), key = lambda x:x[1], reverse= True)}
        count_films = {i:j for i,j in sorted(count_films.items(), key = lambda x:x[1], reverse= True)}
        df = pd.read_csv(ratings_path)
        self.users = list(count_user.keys())[:n_users+1]
        self.films = list(count_films.keys())[:n_films+1]
        rates_matrix = np.zeros((len(self.films), len(self.users)))
        for i_m, m in enumerate(self.films):
            for i_u, u in enumerate(self.users):
                d = df[(df.movieId == m) & (df.userId == u)]
                if d.shape[0] == 0:
                    r = -1
                else:
                    r = d["rating"].values[0]
                rates_matrix[i_m, i_u] = r
        self.rates_matrix = rates_matrix
        self.__remove_empty()
        
    def get_users(self):
        return self.users
    
    def binarize_vector(self,v):
        L = []
        for i in v:
            L.append(1 if i >= 2.5 else 0)
            L.append(1 if i < 2.5 else 0)
        return np.array(L)
    
    def get_subset(self, v):
        t = np.array(v)
        return tuple(np.where(t == 1)[0])
    
    def get_preferences_items(self, user):
        u_rates = self.rates_matrix[:, user]
        others = np.hstack([self.rates_matrix[:, :user], self.rates_matrix[:, user+1:]])
        v = np.array([self.binarize_vector(v) for v in others])
        D = {}
        for k,r in zip(v, u_rates):
            D[tuple(k)] = D.get(tuple(k), []) + [r]
            
        D2 = {}
        for i in D:
            L = D[i]
            if all(i == -1 for i in L):
                continue
            elif all(i != -1 for i in L):
                D2[self.get_subset(i)] = np.mean(L)
            else:
                D2[self.get_subset(i)] = np.mean([i for i in L if i != -1])
                
        items = list(np.arange(max(max(i) for i in D2.keys())))
        prf = Preferences(items)
        for s_1 in D2:
            for s_2 in D2:
                if all(i in s_1 for i in s_2) and all(i in s_2 for i in s_1):
                    continue
                if D2[s_1] > D2[s_2]:
                    prf.add_preference(s_1, s_2)
                elif D2[s_2] > D2[s_1]:
                    prf.add_preference(s_2, s_1)
        return prf
                
    def __remove_empty(self):
        v = np.where(self.rates_matrix.sum(axis = 1) > -self.n_users)[0]
        rates = self.rates_matrix[v, :]