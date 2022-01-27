import numpy as np
import heapq as hq
from typing import Union
import random

class Graph:
    def __init__(self, adjacency_mat: Union[np.ndarray, str]):
        """ Unlike project 2, this Graph class takes an adjacency matrix as input. `adjacency_mat` 
        can either be a 2D numpy array of floats or the path to a CSV file containing a 2D numpy array of floats.

        In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
        elif type(adjacency_mat) == np.ndarray:
            self.adj_mat = adjacency_mat
        else: 
            raise TypeError('Input must be a valid path or an adjacency matrix')
        

    def _load_adjacency_matrix_from_csv(self, path: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=',')

    def construct_mst(self):
        
        #initialize the algorithm at a random node
        all_nodes=set(range(self.adj_mat.shape[0]))
        starting_vertex=random.choice(list(all_nodes))
        visited=set([starting_vertex])
        
        #store edges of mst in a set for now
        mst=set()
        
        # create the priority queue
        # the priority queue stores values (edge_weight, i, (start_node, end_node))
        # i is a counter variable that counts when an edge is added to queue
        #this helps settle "ties", ie, when edges have the same weight
        i=1 
        h=[(z,i,(starting_vertex,y)) for y,z in zip(range(0,self.adj_mat.shape[0]), self.adj_mat[starting_vertex,:]) if z>0]
        hq.heapify(h)
        
        
        while visited != all_nodes:
            print(h)
            edge_weight, visit_order, edge = hq.heappop(h) #pop lowest edge from queue

            if edge[1] not in visited: #if we have not visited the target node yet...
                i+=1 #advance counter variable, b/c were about to add to queue,
                mst=mst.union({(edge_weight,edge)}) #add the edge to the MST,
                visited=visited.union({edge[1]}) #add target node to visited,
                # and add all of its edges to the queue.
                edges=[(z,i, (edge[1],y)) for y,z in zip(range(0,self.adj_mat.shape[1]), self.adj_mat[edge[1],:]) if z>0]
                for e in edges:
                    hq.heappush(h,e)
            
        #convert MST to matrix
        mst_matrix=np.zeros((self.adj_mat.shape[0],self.adj_mat.shape[0]), float)
        for edge in mst:
            x=edge[1][0] # start node
            y=edge[1][1] # target node
            z=edge[0] # edge weight

            mst_matrix[x,y]=z # these will be the same because undirected graphs are symmetric
            mst_matrix[y,x]=z
            
        self.mst=mst_matrix
        
