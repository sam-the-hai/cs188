# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.
        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.
        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        
        # Handle the case when there's no food left
        foodList = newFood.asList()
        if not foodList:
            minFoodDistance = 0
        else:
            minFoodDistance = min(util.manhattanDistance(newPos, food) for food in foodList)
        
        # Calculate ghost distances
        ghostDistances = [util.manhattanDistance(newPos, ghostState.getPosition()) for ghostState in newGhostStates]
        
        # Handle the case when there are no ghosts
        if not ghostDistances:
            minGhostDistance = float('inf')
        else:
            minGhostDistance = min(ghostDistances)
        
        # Calculate a bonus for eating food or capsules
        foodBonus = 0
        if successorGameState.getNumFood() < currentGameState.getNumFood():
            foodBonus = 100  # Bonus for eating food
        
        capsuleBonus = 0
        if len(successorGameState.getCapsules()) < len(currentGameState.getCapsules()):
            capsuleBonus = 200  # Bonus for eating capsules
        
        # Calculate scared ghost bonus
        scaredGhostBonus = 0
        for i, ghostState in enumerate(newGhostStates):
            if ghostState.scaredTimer > 0:
                # If ghost is scared, we want to be closer to it
                scaredGhostBonus += 200 / (1 + util.manhattanDistance(newPos, ghostState.getPosition()))
        
        # Ghost penalty - avoid non-scared ghosts
        ghostPenalty = 0
        for i, ghostState in enumerate(newGhostStates):
            if ghostState.scaredTimer == 0 and ghostDistances[i] <= 2:
                # Heavy penalty for being too close to non-scared ghost
                ghostPenalty = -500 / (ghostDistances[i] + 0.1)
        
        # Penalty for being far from food
        # foodDistancePenalty = 0 
        foodDistancePenalty = -2 * minFoodDistance
        
        # Calculate final score
        score = successorGameState.getScore() + foodDistancePenalty + foodBonus + capsuleBonus + scaredGhostBonus + ghostPenalty

        # General ghost distance factor (only for non-scared ghosts)
        if minGhostDistance < float('inf'):
            ghostDistanceFactor = min(50, minGhostDistance) # Cap the bonus
            score += ghostDistanceFactor
        
        return score


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        # https://en.wikipedia.org/wiki/Minimax
        def minimax(agentIndex: int, depth: int, state: GameState) -> float:
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            numAgents = state.getNumAgents()
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth

            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(state)

            if agentIndex == 0:  # Pacman (maximize)
                return max(minimax(nextAgent, nextDepth, state.generateSuccessor(agentIndex, action)) for action in legalActions)
            else:  # Ghosts (minimize)
                return min(minimax(nextAgent, nextDepth, state.generateSuccessor(agentIndex, action)) for action in legalActions)

        # Get the best action by comparing minimax values
        bestScore = float('-inf')
        bestAction = None
        for action in gameState.getLegalActions(0):
            value = minimax(1, 0, gameState.generateSuccessor(0, action))
            if value > bestScore:
                bestScore = value
                bestAction = action

        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        def alphaBeta(agentIndex: int, depth: int, state: GameState, alpha: float, beta: float) -> tuple[float, str]:
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state), None

            numAgents = state.getNumAgents()
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth

            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(state), None

            if agentIndex == 0:  # Pacman (maximize)
                value = float('-inf')
                bestAction = None
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    score, _ = alphaBeta(nextAgent, nextDepth, successor, alpha, beta)
                    if score > value:
                        value = score
                        bestAction = action
                    if value > beta:
                        break  # Beta cutoff
                    alpha = max(alpha, value)
                return value, bestAction
            else:  # Ghosts (minimize)
                value = float('inf')
                bestAction = None
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    score, _ = alphaBeta(nextAgent, nextDepth, successor, alpha, beta)
                    if score < value:
                        value = score
                        bestAction = action
                    if value < alpha:
                        break  # Alpha cutoff
                    beta = min(beta, value)
                return value, bestAction

        # Pacman is agentIndex = 0
        _, action = alphaBeta(0, 0, gameState, float('-inf'), float('inf'))
        return action
        
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
