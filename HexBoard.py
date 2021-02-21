# Hex board skeleton + .get_allempty()

class HexBoard:
    BLUE = 1  # value take up by a position
    RED = 2
    EMPTY = 3

    def __init__(self, board_size):
        """Constructor, set board size"""
        self.board = {}
        self.size = board_size
        self.game_over = False  # state of game over
        for x in range(board_size):
            for y in range(board_size):
                self.board[x, y] = HexBoard.EMPTY
        
    def is_game_over(self):
        """Check if it's game over"""
        return self.game_over

    def is_empty(self, coordinates):
        """Check if position is empty"""
        return self.board[coordinates] == HexBoard.EMPTY

    def is_color(self, coordinates, color):
        """Check if position contain certain color 1/2"""
        return self.board[coordinates] == color

    def get_color(self, coordinates):
        """Read color of a position"""
        if coordinates == (-1, -1):
            return HexBoard.EMPTY
        return self.board[coordinates]

    def place(self, coordinates, color):
        """
        Place a piece of color at a position, make move? Will update game over state.
        Check condition if it's not game over AND position is empty
        """
        if not self.game_over and self.board[coordinates] == HexBoard.EMPTY:
            self.board[coordinates] = color  # update the color
            if self.check_win(HexBoard.RED) or self.check_win(HexBoard.BLUE):  # check win for either color
                self.game_over = True  # if check win is true, one side win, then update game over state to true
        
    @staticmethod
    def get_opposite_color(current_color):
        """return opposite color. what is the purpose?"""
        if current_color == HexBoard.BLUE:
            return HexBoard.RED
        return HexBoard.BLUE

    def get_neighbors(self, coordinates):
        """Return a list of valid neighbor coordinates from a position"""
        (cx, cy) = coordinates
        neighbors = []
        if cx - 1 >= 0:
            neighbors.append((cx - 1, cy))
        if cx + 1 < self.size:
            neighbors.append((cx + 1, cy))
        if cx - 1 >= 0 and cy + 1 <= self.size - 1:
            neighbors.append((cx - 1, cy + 1))
        if cx + 1 < self.size and cy - 1 >= 0:
            neighbors.append((cx + 1, cy - 1))
        if cy + 1 < self.size:
            neighbors.append((cx, cy + 1))
        if cy - 1 >= 0:
            neighbors.append((cx, cy-1))
        return neighbors

    def border(self, color, move):
        """Check if a move is the right color reaching the right border, blue1-x, red2-y"""
        (nx, ny) = move
        return (color == HexBoard.BLUE or color == HexBoard.RED) and ny == self.size - 1

    def traverse(self, color, move, visited):
        """Move is the target position"""
        if not self.is_color(move, color) or (move in visited and visited[move]):
            return False  # check if move position do NOT contain my color AND is NOT in visited
        if self.border(color, move):
            return True  # check if the move is reaching border
        visited[move] = True  # update position in visited (move history)
        for n in self.get_neighbors(move):  # check all neigbour positions if the move passes all checks above
            if self.traverse(color, n, visited):
                return True
        return False

    def check_win(self, color):
        """Check win condition"""
        for i in range(self.size):
            if color == HexBoard.BLUE:
                move = (0, i)  # for blue, move rightward (0,0), (0,1), (0,2), ... = start check from one border
            else:
                move = (i, 0)  # for red, move downward (0,0), (1,0), (2,0), ...
            # If true in traverse, return win.
            # Note that traverse will return check if right color reach the right border
            if self.traverse(color, move, {}):
                return True
        return False

    def print(self):
        print("   ", end="")
        for y in range(self.size):
            print(chr(y + ord('a')), "", end="")  # print x axis id
        print("")
        print(" -----------------------")
        for y in range(self.size):
            print(y, "|", end="")  # print y axis id
            for z in range(y):
                print(" ", end="")  # print space
            for x in range(self.size):
                piece = self.board[x, y]  # read position
                if piece == HexBoard.BLUE:
                    print("b ", end="")  # end=print without newline
                elif piece == HexBoard.RED:
                    print("r ", end="")
                else:
                    if x == self.size:
                        print("-", end="")
                    else:
                        print("- ", end="")
            print("|")  # print '|' and new line by default
        print("   -----------------------")

    # return list of empty positions
    def get_allempty(self):
        """Return a list of empty positions in current board, same as movelist."""
        return [k for k, v in self.board.items() if v == 3]  # 3 = EMPTY

    def convert_key(self):
        """Return a key (str) that represent board positions, unique"""
        key = "" # initiate
        for y in range(self.size):
            for x in range(self.size):
                key += str(self.board[x, y])  # read piece state {1/2/3}
        return key