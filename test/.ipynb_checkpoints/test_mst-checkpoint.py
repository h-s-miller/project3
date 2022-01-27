# write tests for bfs
import pytest
import numpy as np
from mst import Graph
from sklearn.metrics import pairwise_distances


def check_mst(adj_mat: np.ndarray, 
              mst: np.ndarray, 
              expected_weight: int, 
              allowed_error: float = 0.0001):
    """ Helper function to check the correctness of the adjacency matrix encoding an MST.
        Note that because the MST of a graph is not guaranteed to be unique, we cannot 
        simply check for equality against a known MST of a graph. 

        Arguments:
            adj_mat: Adjacency matrix of full graph
            mst: Adjacency matrix of proposed minimum spanning tree
            expected_weight: weight of the minimum spanning tree of the full graph
            allowed_error: Allowed difference between proposed MST weight and `expected_weight`
    """
    # test 1: the MST total weight should be within small error of expected total weight (written by TAs)
    def approx_equal(a, b):
        return abs(a - b) < allowed_error

    total = 0
    for i in range(mst.shape[0]):
        for j in range(i+1):
            total += mst[i, j]
    assert approx_equal(total, expected_weight), 'Proposed MST has incorrect expected weight'
    
    # test 2: the number of edges in a minimum spanning tree = N-1 where N=number of nodes
    assert (np.count_nonzero(mst)/2)==(adj_mat.shape[0]-1), 'Proposed MST has incorrect number of edges'
    
    # test 3: the minimum weight edge of the graph must exist in the MST 
    assert np.amin(mst) == np.amin(adj_mat), 'Proposed MST does not satisfy minimum edge weight property'
    
def test_mst_small():
    """ Unit test for the construction of a minimum spanning tree on a small graph """
    file_path = './data/small.csv'
    g = Graph(file_path)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 8)


def test_mst_single_cell_data():
    """
    Unit test for the construction of a minimum spanning tree using 
    single cell data, taken from the Slingshot R package 
    (https://bioconductor.org/packages/release/bioc/html/slingshot.html)
    """
    file_path = './data/slingshot_example.txt'
    # load coordinates of single cells in low-dimensional subspace
    coords = np.loadtxt(file_path)
    # compute pairwise distances for all 140 cells to form an undirected weighted graph
    dist_mat = pairwise_distances(coords)
    g = Graph(dist_mat)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 57.263561605571695)


def test_mst_symmetric(allowed_error: float = 0.0001):
    """
     Unit test to check that the MST is symmetric, as it should be for undirected graph.
    Note: If  matrix A is symmetric, then A-A^T = 0 
    """
    file_path = './data/small.csv'
    g = Graph(file_path)
    g.construct_mst()
    assert np.all(np.abs(g.mst-g.mst.T)<allowed_error)==True # A-A^T ==0 

