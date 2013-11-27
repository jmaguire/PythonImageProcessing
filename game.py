





class TicTacToeState:
    roles = ['x','o']
    
    def toHash(self):
        board = (tuple(self.board[0]),tuple(self.board[1]),tuple(self.board[2]))
        return (board, self.role)
        
    def __init__(self, board = None, role = None):

        if board == None or role == None:
            self.board = [[None]*3 for j in range(3)]
            self.role = 'x' ##human is x. starts with human
        else:
            self.board = list(board)
            self.role = role
  
    ## action is a i,j location
    ## we only need to check row,col and diag that the action is in
    
    def copy(self):
        board = list([list(self.board[0]),list(self.board[1]),list(self.board[2])])
        return TicTacToeState(board,self.role)
        
    ## this is because we can only win on a action besides the last action (always a draw)
    def isGoalAndScore(self):
        def checkEqual(list):
            if list == tuple([TicTacToeState.roles[0]])*3 or list == tuple(TicTacToeState.roles[1]*3): return True
    

        
        ## check win conditions
        
        ##check row
        for i in range(3):
            row = tuple(self.board[i])
            if checkEqual(row): return (True,100) 
        ##check column
        for i in range(3):
            col = tuple([self.board[0][i],self.board[1][i],self.board[2][i]])
            if checkEqual(col): return (True,100) 
          
        ## check diagonals
        diag_top_left = tuple([self.board[0][0],self.board[1][1],self.board[2][2]])
        diag_top_right = tuple([self.board[0][2],self.board[1][1],self.board[2][0]])
       
        if checkEqual(diag_top_left) or checkEqual(diag_top_right): return (True,100)
        
        ## Board full but no win?
        freeSpaces = sum([1 for col in self.board for row in col if row == None])
        if freeSpaces == 0: return (True,0)

        return (False,0)
        
     ## this is because we can only win on a action besides the last action (always a draw)
    def isWin(self):
        def checkEqual(list):
            if list == tuple(['o'])*3 or list == tuple('o'*3): return True
    
        ## check win conditions
        
        ##check row
        for i in range(3):
            row = tuple(self.board[i])
            if checkEqual(row): return True
        ##check column
        for i in range(3):
            col = tuple([self.board[0][i],self.board[1][i],self.board[2][i]])
            if checkEqual(col): return True
          
        ## check diagonals
        diag_top_left = tuple([self.board[0][0],self.board[1][1],self.board[2][2]])
        diag_top_right = tuple([self.board[0][2],self.board[1][1],self.board[2][0]])
       
        if checkEqual(diag_top_left) or checkEqual(diag_top_right): return True
        
        return False
    
    def isLose(self):
        
        def checkEqual(list):
            if list == tuple(['x'])*3 or list == tuple('x'*3): return True
    
        ## check win conditions
        
        ##check row
        for i in range(3):
            row = tuple(self.board[i])
            if checkEqual(row): return True
        ##check column
        for i in range(3):
            col = tuple([self.board[0][i],self.board[1][i],self.board[2][i]])
            if checkEqual(col): return True
          
        ## check diagonals
        diag_top_left = tuple([self.board[0][0],self.board[1][1],self.board[2][2]])
        diag_top_right = tuple([self.board[0][2],self.board[1][1],self.board[2][0]])
       
        if checkEqual(diag_top_left) or checkEqual(diag_top_right): return True
        
        return False

    def isGoal(self):
        return self.isGoalAndScore()[0]
        
    def getScore(self):
        if self.isWin(): return 100
        if self.isLose(): return -100
        return 0
        #return self.isGoalAndScore()[1]
    
    # Only the computer uses this. So define in terms of computer
    # def isLose(self):
        # if self.getScore() == 100 and self.role != 'o': return True
        # return False
        
    # def isWin(self):
        # if self.getScore() == 100 and self.role == 'o': return True 
        # return False
    
    def getOtherRole(self):
        return TicTacToeState.roles[0] if self.role != TicTacToeState.roles[0] else TicTacToeState.roles[1]
    
    
    ## action is (i,j) pair
    ## returns new states
    def generateSuccessor(self,action):
        if self.board[action[0]][action[1]] != None:
            print action
            raise Exception('Illegal Move')
        else:
            board = list([list(self.board[0]),list(self.board[1]),list(self.board[2])])
            board[action[0]][action[1]] = self.role
            return TicTacToeState(board,self.getOtherRole())
            
    # returns list of (action)
    def getLegalActions(self):
        otherRole = self.getOtherRole()
        actions = [(row,col) for row in range(3) for col in range(3) if self.board[row][col] == None]
        return actions
        

            