# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 16:57:16 2018

@author: Becca
"""
import numpy as np
import tsne_new as tsne
import matplotlib.pyplot as plt
from collections import deque
import networkx as nx
from sklearn.cluster import DBSCAN

from sklearn.cluster import AffinityPropagation
from networkx.algorithms.community import greedy_modularity_communities
from sklearn.decomposition import LatentDirichletAllocation

import community


##############################################################################
# Functions for creating stochastic block model
##############################################################################
def create_network(node_count, partition_lens=[]):
    '''
    Returns the adjacency matrix for a random stocastic block model with a given 
    number of nodes and partition.   
    '''
    
    # Create partition
    if len(partition_lens) == 0:
        nodes_used = 0
        while nodes_used < node_count:
            if node_count - nodes_used < 40:
                size = node_count - nodes_used
            else:
                size = np.random.randint(20, node_count/2)
            partition_lens.append(size)
            nodes_used += size
            
    # Create communities
    total = 0
    partition_start = []
    for i in partition_lens:
        partition_start.append(total)
        total += i
               
    num_partitions = len(partition_lens)
    communities = [list(range(i,i+j)) for i,j in zip(partition_start, partition_sizes)]
    
    # Create list of which community each nodes belongs to    
    node_comm = []
    for i in range(num_partitions):
        node_comm.append([i]*partition_lens[i])
    node_ids =[item for sublist in node_comm for item in sublist]
            
    A = create_adj(node_count, node_ids)

    return A, communities

def create_adj(n, comm_list):
    """
        Returns an adjacency matrix representing a random network based on 
        a set of communities and the probabilities that they are connected.       
    """
    same_prob = .9
    diff_prob = .2
    A = np.zeros((n,n))
    for i in range(1, n):
        for j in range(i):
            prob = np.random.rand()
            
            if comm_list[i] == comm_list[j]:
                if prob < same_prob:
                    A[i][j]=1
                    A[j][i]=1
            else:
                if prob < diff_prob:
                    A[i][j]=1
                    A[j][i]=1        
    return A

##############################################################################
# Testing Functions
# Returns number between 0 and 1, where 1 means predicted partitions are the 
# same as the actual
##############################################################################
    
def nmi(prediction, actual, nodes):
    '''
    Normalized Mutual Information
    '''
    
    predicted_partitions = len(prediction)
    actual_partitions = len(actual)
        
   # Create C
    C = np.zeros([actual_partitions,predicted_partitions])
    for i in range(actual_partitions):
        for node in actual[i]:
            for j in range(predicted_partitions):                 
                if node in prediction[j]:
                    C[i,j] += 1
    
    # Get Ci. and Cj.
    row_sums = np.zeros(actual_partitions)
    column_sums = np.zeros(predicted_partitions)
    
    for i in range(actual_partitions):
        row_sums[i] = sum(C[i])
        
    for j in range(predicted_partitions):
        column_sums[j] = sum(C[:,j])
    
    num = 0
    for j in range(predicted_partitions):
        for i in range(actual_partitions):
            if C[i,j] != 0:
                num += C[i][j] * np.log( C[i][j] * nodes / (row_sums[i] * column_sums[j]) )
            
    num = -2 * num
    
    den = 0

    for i in range(actual_partitions):
        if row_sums[i] != 0:
            den += row_sums[i] * np.log(row_sums[i]/nodes)
    for j in range(predicted_partitions):
        if column_sums[j] != 0:
            den += column_sums[j] * np.log(column_sums[j]/nodes)

    if den == 0:
        return 0
    else:
        return num/den

def f_score(prediction, actual):
    '''
    F-scores
    '''
    
    pred = set(tuple(i) for i in prediction)
    act = set(tuple(i) for i in actual)
    precision = len( act.intersection(pred )) / len(pred)
    recall =  len( act.intersection(pred )) / len(act)

    if precision == 0.0 and recall == 0.0:
        score = 0
    else:
        score = 2 * ( (precision*recall) / (precision + recall) )
    return score
    

def jaccard(prediction, actual):
    '''
    Jaccard
    '''
    pred = set(tuple(i) for i in prediction)
    act = set(tuple(i) for i in actual)
    
    num =  len( act.intersection(pred ))    
    den = len(act) + len(pred) - num
    
    return num/den
    
 
def AC(prediction, actual):
    '''
    Relative error of predicting number of communities
    '''
    ac = 1-abs(len(actual)-len(prediction))/len(actual)
    return ac
    
def test(prediction, actual, nodes):
    '''
    Runs all four testing algorithms
    '''
    
    #print("p",prediction)

    
    return [nmi(prediction, actual, nodes),
           AC(prediction,actual),
           f_score(prediction, actual),
           jaccard(prediction, actual)]


###############################################################################
# Comparison Algorithms
###############################################################################

def AP(A, actual, nodes):
    '''
    Affinity Propogation
    '''
    clustering = AffinityPropagation().fit(A)
    labels = clustering.labels_
    ap_communities  = [[i for i, x in enumerate(labels) if x == j] for j in range(max(labels)+1)]
    
    return test(ap_communities, actual, nodes)

def CNM(A, actual, nodes):
    '''
    Clauset-Newman-Moore greedy modularity maximization
    '''
    G=nx.from_numpy_matrix(A)
    c = list(greedy_modularity_communities(G))
    cnm_communities = [sorted(i) for i in c]

    return test(cnm_communities, actual, nodes)

    

def louvain(A, actual, nodes):
    '''
    Louvain
    '''
    G=nx.from_numpy_matrix(A)

    partition = community.best_partition(G)
    node_list = list(partition.values())
    communities = [[i for i, x in enumerate(node_list) if x==j] for j in range(max(node_list))]

    return test(communities, actual, nodes)

def LDA(A, actual,nodes):
    '''
    Latent Dirichlet Allocation 
    '''
    lda = LatentDirichletAllocation().fit(A)
    
    print(lda.transform(A))

    #return test(cnm_communities, actual)


def comparisons(A, actual,nodes):
    

    cnm = ['CNM', CNM(A, actual,nodes)]
    
    louv = ["Louvain", louvain(A,actual,nodes)]
    
    ap = ["AP Results", AP(A, actual,nodes)]
    
  #  print("DPA")
  #  DP(A,actual)
    
  #  print("LDA")
  #  LDA(A,actual)
  
  #  print("PCL-DC")
  #  PCL(A,actual)
    
  #  print("Block-LDA")
  #  BLDA(A,actual)
  
  #  print("AE")
  #  AE(A,actual)
    
    return [cnm, louv, ap]


##############################################################################
# TSNE related functions
##############################################################################  
    
def graph(n_communities, X, communities):
    '''
    Graph the communities by partition using TSNE output
    '''
    
    fig = plt.figure() 
    ax = plt.subplot(111)
    
    for j in range(n_communities):
        ax.scatter(X[communities[j], 0],
                          X[communities[j], 1], 
                          s=5, label=j)
            
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

def dbscan(X,eps=15, min_samples=10):
    '''
    Performs DBScan clustering to identify clusters after TSNE
    '''
    
    db = DBSCAN(eps, min_samples).fit(X)
    p_ids = db.labels_


    # Number of clusters in labels, ignoring noise if present.
    # TODO: only some values are -1
    if np.all(p_ids == -1):
        n_clusters = 1
        p_ids = np.zeros(len(p_ids),dtype=np.int32)
      #  p_ids = 
    else:
        n_clusters = len(set(p_ids)) - (1 if -1 in p_ids else 0)

    p_communities = [[i for i, x in enumerate(p_ids) if x == j] for j in range(n_clusters)]
    #p_communities not useful

    return p_ids, n_clusters, p_communities

  
def bfs(A, communities, community_i=[], eps=15, min_samples=10):   
    '''
    Runs TSNE on the nodes in a breadth-first search manner until TSNE returns
    only one community for each of possible community
    '''
    
    # Graph the data after TSNE
    X=tsne.tsne(A,2,2,16)
    n_communities = len(communities)
    
    if len(community_i) == 0:
        graph(n_communities, X, communities)
    else:
        graph(n_communities, X, community_i)
    # Run DBScan to identify clusters
    p_ids, n_clusters, p_communities = dbscan(X, eps, min_samples)    # p_ids is np array, p_communities is list of lists 
    
    # Breadth-first Search
    clusters = deque()
    prediction=[]
    nodes_removed = 0
    
    # Initialize deque with first row of clusters
    for i in p_communities:
        clusters.append(i)
        
    # For each cluster, run TSNE and add new clusters
    while clusters:
        # Get cluster and A to test
        test_cluster = clusters.popleft() #test_cluster is list
        test_A = A[test_cluster][:,test_cluster]

        # Run TSNE on cluster and add new clusters to deque      
        Y=tsne.tsne(test_A,2,2,16)
        p_ids, n_clusters, p_communities = dbscan(Y, eps, min_samples)
        
        unique, counts = np.unique(p_ids, return_counts=True)
        counts = dict(zip(unique, counts))
        if -1 in counts: 
            nodes_removed += counts[-1]

        new_clusters = [np.flatnonzero(p_ids== j) for j in range(p_ids[-1]+1)]
                
        for cluster in new_clusters:
            indices = [test_cluster[i] for i in cluster]
            
            if len(new_clusters) == 1:
                prediction.append(indices)
            else:

                clusters.append(indices)
    print("Number of nodes removed:", nodes_removed)       
    print("Final Community Prediction after TSNE")
    graph(len(prediction),X, prediction)

    
    return prediction
 

if __name__ == "__main__":
    
    # Create network
    partition_sizes= [20,50,50, 80,110,120,70,100,100,500]
    nodes = sum(partition_sizes)

    A, communities = create_network(nodes, partition_sizes)
    prediction = bfs(A, communities)  

    
    print(['NMI', 'AC', 'F-score', 'Jaccard'])
    test(prediction, communities, nodes)
       
    comparisons(A, communities, nodes)
    
    
    
    