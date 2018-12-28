# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 19:38:57 2018

Import fb data

@author: Becca
"""

import networkx as nx
import csv
import iterative_tsne as tsne
import numpy as np



outfile = "test.csv"
# Import edge list and create adjacency matrix

fh=open("facebook/1684.edges", 'rb')
G=nx.read_edgelist(fh)
fh.close()
A = nx.to_numpy_matrix(G)
A = np.asarray(A)
nodes = len(G)


# Import communities
with open('facebook/1684.circles') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='\t', quotechar='\'')
    
    partition = [row[1:] for row in csv_reader]
    communities = partition.copy()
    for index, cols in enumerate(partition):
        communities[index] = [int(cols[i]) for i in range(len(cols))]
        
nodes_list = list(G.nodes())

community_i= []
for block in partition:
    community = []
    for node in block:
        if node in nodes_list:
            community.append(int(nodes_list.index(node)))
    community_i.append(community)
    

# Run tsne and compare
prediction = tsne.bfs(A, communities, community_i, eps=8, min_samples=4)  
p_results = tsne.test(prediction, communities, nodes)  
c_results = tsne.comparisons(A, communities, nodes)

print(['NMI', 'AC', 'F-score', 'Jaccard'])
print(['Ours', p_results])
for results in c_results:
    print([results])

#with open(outfile, 'w', newline='') as csv_file:
#    csv_writer = csv.writer(csv_file, delimiter='\t')
#     
#    csv_writer.writerow(['NMI', 'AC', 'F-score', 'Jaccard'])
#    csv_writer.writerow(['Ours', p_results])
#    for results in c_results:
#        csv_writer.writerow([results])

