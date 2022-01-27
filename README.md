! [BuildStatus] (https://github.com/h-s-miller/project3/workflows/project3/badge.svg?event=push)
# Project 3
Minimum Spanning Trees

# Assignment Overview
The purpose of this assignment is to implement Prim's algorithm, a non-trivial greedy algorithm used to construct minimum spanning trees. 

# Assignment Tasks

## Coding Assessment
* [DONE] Complete the `Graph.construct_mst` method found in the mst/graph.py 
	* All necessary modules have already been imported. We encourage you to not rely on any other dependencies (e.g. networkx). 

## Software Development Assessment

* [DONE] Add more assertions to the `check_mst` function in test/test_mst.py 
* [DONE] Write at least one more unit test (in the test_mst.py file) for your construct_mst implementation
* Two unit tests have already been provided. 
    * The first is a small graph of four nodes
    * The second is a larger graph of 140 single cells, projected onto a lower dimensional subspace
	
* Automate Testing with [Github Actions](https://docs.github.com/en/actions)

	See blogposts below on helping set up Github actions with pytest:
	
	* [post 1](https://blog.dennisokeeffe.com/blog/2021-08-08-pytest-with-github-actions)
	* [post 2](https://mattsegal.dev/pytest-on-github-actions.html)
	* Add "! [BuildStatus] (https://github.com/ < your-github-username > /Project3/workflows/Project3/badge.svg?event=push)" (update link and remove spaces) to the beginning of your README file
	* Also refer to previous assignment for more in-depth help with GitHub actions

	Ensure that the Github actions complete the following:
	* runs pytest

# Getting Started
To get started you will need to fork this repository onto your own Github account. Work on the codebase from your own repo and commit changes. 

The following packages will be needed:
    - numpy
    - scikit-learn
    - pytest
    - heapq [optional, but highly encouraged]

# Completing the assignment
Make sure to push all your code to Github, ensure that your unit tests are correct, and submit a link to your Github through the Google classroom assignment.

# Grading

## Code (6 points)
* Minimum spanning tree construction works correctly (6)
    * Correct implementation of Prim's algorithm (4)
    * Produces expected output on small graph (1) 
    * Produces expected output on single cell data (1) 

## Unit tests (3 points)
* Added additional checks in `check_mst` to ensure correctness of your implementation (2)
* Added effective unit tests (1)

## Style (1 points)
* Readable code with clear comments and method descriptions
