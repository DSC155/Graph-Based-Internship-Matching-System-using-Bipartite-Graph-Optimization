import networkx as nx
import pandas as pd
from collections import deque

def parse_skills(skills_str):
    
    if pd.isna(skills_str):
        return set()
    return set([s.strip().lower() for s in str(skills_str).split(',') if s.strip()])

def compute_similarity_matrix(students_df, companies_df, skills_col='skills'):
    
    students_skills = [parse_skills(x) for x in students_df[skills_col]]
    companies_skills = [parse_skills(x) for x in companies_df[skills_col]]
    sim = pd.DataFrame(0.0, index=range(len(students_skills)), columns=range(len(companies_skills)))
    for i, sset in enumerate(students_skills):
        for j, cset in enumerate(companies_skills):
            if not sset and not cset:
                score = 0.0
            else:
                inter = len(sset & cset)
                union = len(sset | cset)
                score = inter / union if union > 0 else 0.0
            sim.at[i, j] = round(score, 4)
    return sim

def greedy_weighted_matching(sim_matrix, threshold=0.0):
   
    edges = []
    for i in sim_matrix.index:
        for j in sim_matrix.columns:
            w = float(sim_matrix.at[i, j])
            if w >= threshold:
                edges.append((w, i, j))
    edges.sort(reverse=True)  
    matched_students = set()
    matched_companies = set()
    matches = []
    for w, i, j in edges:
        if i in matched_students or j in matched_companies:
            continue
        matched_students.add(i)
        matched_companies.add(j)
        matches.append((i, j, w))
    return matches

import networkx as nx

import networkx as nx

def hopcroft_karp_matching(sim, threshold=0.0):
  
    G = nx.Graph()
    num_students, num_companies = sim.shape

    
    left_nodes = [f"S_{i}" for i in range(num_students)]
    right_nodes = [f"C_{j}" for j in range(num_companies)]
    G.add_nodes_from(left_nodes, bipartite=0)
    G.add_nodes_from(right_nodes, bipartite=1)

    
    for i in range(num_students):
        for j in range(num_companies):
            if sim.iloc[i, j] >= threshold:
                G.add_edge(f"S_{i}", f"C_{j}", weight=float(sim.iloc[i, j]))

    
    if len(G.edges) == 0:
        return []

    # Use NetworkX’s optimized Hopcroft–Karp algorithm
    matching = nx.bipartite.maximum_matching(G, top_nodes=left_nodes)

    
    matches = []
    for u, v in matching.items():
        if u in left_nodes:  # keep only left-to-right mappings
            matches.append((int(u.split('_')[1]), int(v.split('_')[1]), G[u][v]['weight']))

    return matches



def hungarian_matching(sim_matrix):
   
    try:
        from scipy.optimize import linear_sum_assignment
    except Exception as e:
        raise ImportError("scipy is required for hungarian_matching: pip install scipy") from e

    import numpy as np
    
    cost = 1.0 - sim_matrix.values  # sim in [0,1], so cost in [0,1]
    row_ind, col_ind = linear_sum_assignment(cost)
    matches = []
    for r, c in zip(row_ind, col_ind):
        
        matches.append((int(r), int(c), float(sim_matrix.iat[r, c])))
    return matches
