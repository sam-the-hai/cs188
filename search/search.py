# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Directions
from typing import List

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()




def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    from util import Stack

    # Initialize the frontier with the start state and an empty path
    frontier = Stack()
    frontier.push((problem.getStartState(), []))
    visited = set()

    # Process nodes in the frontier
    while not frontier.isEmpty():
        # Get the next node from the frontier
        state, path = frontier.pop()

        # If the node is a goal state, return the path
        if problem.isGoalState(state):
            return path
        
        # If the node has already been visited, skip it
        if state in visited:
            continue

        # Add the node to the visited set
        visited.add(state)

        # Expand the node and add its successors to the frontier
        for successor, action, cost in problem.getSuccessors(state):
            new_path = path + [action]
            frontier.push((successor, new_path))

    # If the frontier is empty and no goal state was found, return an empty path
    return []

def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""
    from util import Queue

    # Initialize the frontier with the start state and an empty path
    frontier = Queue()
    frontier.push((problem.getStartState(), []))
    visited = set()

    # Process nodes in the frontier
    while not frontier.isEmpty():
        # Get the next node from the frontier
        state, path = frontier.pop()

        # If the node is a goal state, return the path
        if problem.isGoalState(state):
            return path
        
        # If the node has already been visited, skip it
        if state in visited:
            continue

        # Add the node to the visited set
        visited.add(state)

        # Expand the node and add its successors to the frontier
        for successor, action, cost in problem.getSuccessors(state):
            new_path = path + [action]
            frontier.push((successor, new_path))

    # If the frontier is empty and no goal state was found, return an empty path
    return []

def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""
    from util import PriorityQueue

    # Initialize the frontier with the start state and zero cost and an empty path and priority 0
    frontier = PriorityQueue()
    frontier.push((problem.getStartState(), 0, []), 0)
    visited = set()
    
    # Process nodes in the frontier
    while not frontier.isEmpty():
        # Get the next node from the frontier
        state, cost, path = frontier.pop()

        # If the node is a goal state, return the path
        if problem.isGoalState(state):
            return path
        
        # If the node has already been visited, skip it
        if state in visited:
            continue
        
        # Add the node to the visited set
        visited.add(state)

        # Expand the node and add its successors to the frontier
        for successor, action, stepCost in problem.getSuccessors(state):
            new_path = path + [action]
            new_cost = cost + stepCost
            frontier.push((successor, new_cost, new_path), new_cost)

    # If the frontier is empty and no goal state was found, return an empty path
    return []

def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    from util import PriorityQueue

    # Initialize the frontier with the start state and zero cost and an empty path and priority 0
    frontier = PriorityQueue()
    frontier.push((problem.getStartState(), 0, []), 0)
    
    # Keep track of best cost to reach each state
    best_costs = {problem.getStartState(): 0}

    # Process nodes in the frontier
    while not frontier.isEmpty():
        # Get the next node from the frontier
        state, cost, path = frontier.pop()
        
        # If the node is a goal state, return the path
        if problem.isGoalState(state):
            return path
        
        # If the node has already been visited, skip it
        if state in best_costs and cost > best_costs[state]:
            continue

        # Expand the node and add its successors to the frontier
        for successor, action, stepCost in problem.getSuccessors(state):
            new_path = path + [action]
            new_cost = cost + stepCost

            # Update the best cost for the successor if it's better
            if successor not in best_costs or new_cost < best_costs[successor]:
                best_costs[successor] = new_cost
                priority = new_cost + heuristic(successor, problem)
                frontier.push((successor, new_cost, new_path), priority)

    # If the frontier is empty and no goal state was found, return an empty path
    return []

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
