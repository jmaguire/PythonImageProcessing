import random
import math

class MinimaxAgent:
  """
    Your minimax agent (problem 1)
  """
  
  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves. 

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    
      gameState.isWin():
        Returns True if it's a winning state
    
      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue
    """

    def printBoard(board):
        board = list([list(board[0]),list(board[1]),list(board[2])])
        board[0] = [x + ' ' if x!=None else '[]' for x in board[0]]
        board[1] = [x + ' ' if x!=None else '[]' for x in board[1]]
        board[2] = [x + ' ' if x!=None else '[]' for x in board[2]]
        print
        print board[0][0],board[0][1],board[0][2]
        print board[1][0],board[1][1],board[1][2]
        print board[2][0],board[2][1],board[2][2]
        print
    
    
    ## Computer Move
    def maxScore(state,alpha,beta,depth):
        if state.isGoal(): 
            return state.getScore() ## computer score
        if depth == 0:
            print 'depthlim'
            return state.getScore()
        actions = state.getLegalActions()
        random.shuffle(actions)
        value = float("-inf")
        for action in actions:
            nextState = state.generateSuccessor(action)
            value = max(value,minScore(nextState,alpha,beta,depth - 1))
            if value >= beta: return value 
            alpha  = max(value,alpha)
        return value

    ## Opponent Move
    def minScore(state,alpha,beta,depth):
        if state.isGoal():
            return state.getScore()     
            
        actions = state.getLegalActions()
        random.shuffle(actions)
        value = float("inf")
        successors = [state.generateSuccessor(move) for move in actions]
        for nextState in successors:
            value = min(value,maxScore(nextState,alpha,beta,depth))
            if value <= alpha: return value
            beta = min(value,beta)
        return value
    
    
    


    self.depth = 5
    self.cache = {}
    actionScore = float("-inf")
    actions = gameState.getLegalActions()
    value = float("-inf")
    alpha , beta = float("-inf") , float("inf")
    for action in gameState.getLegalActions():        
        nextState = gameState.generateSuccessor(action)
        if nextState.isWin(): ##isWin = does computer Win
            return action
        value = max(value,minScore(nextState, alpha,beta, self.depth - 1))
        if value == 100: 
            actionScore = value
            bestAction = action
            print 'Woopie!'
            break;
        if value > actionScore:
            actionScore = value
            bestAction = action
    print actionScore
    return bestAction