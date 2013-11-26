from game import TicTacToeState
from minimax import MinimaxAgent
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

gameState = TicTacToeState()
while(gameState.isGoal() is False):
    print gameState.role
    printBoard(gameState.board)
    if gameState.role == 'x':
        move = raw_input('Enter Move x,y:')
        move = move.split(',')
        action = [int(move[0]),int(move[1])]
    else:
        player = MinimaxAgent()
        action = player.getAction(gameState)
    print 'action chosen', action
    gameState = gameState.generateSuccessor(action)
print 'Game Over'
printBoard(gameState.board)

