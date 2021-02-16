'''
Script to repeat n games.

How to use:
Provide input to set up the game.

Parameters:
games to play, board size, player names, print boards or not

Output:
All game results is stored in dict games_result.
Data stored: endstate, winner, turns, elapsed_time
To access individual game e.g. games_result['games']['11']

Q: How to change player decison?
A: Change the value assigned to move.
There are several options (random move, manual move, minimax).
But user has to modify corresponding code.

Missing feature:
unMove() to explore games further
'''

import HexBoard
import Minimax
import random
from time import perf_counter 

# set parameters
games_limit = int(input('Enter number of games you want to play? (integer > 0)'))
board_size = int(input('Enter board size? (integer > 1)'))
p1_id = input('Enter ID for P1 (piece=b)?')
p2_id = input('Enter ID for P2 (piece=r)?')
is_print_endgame = int(input('Do you want to show end state? \n1-True, 0-False'))
is_print_midgame = int(input('Do you want to show midgame? Lengthy printout alert. \n1-True, 0-False'))

print('\nP1: {} (b) and P2: {} (r) will play on {}x{} Hex Board for {} rounds.'.format(p1_id, p2_id, board_size, board_size, games_limit))
print('P1 and P2 will move first in odd and even games respectively.\n')

# initialize games data
games_result = {'p1': p1_id,
                'p2': p2_id,
                'games': {}}

for game_i in range(1, games_limit+1):
    
    game = HexBoard.HexBoard(board_size) # initialize board
    assert game.is_game_over() == False
    turn = 0 # reset turn count
    
    if game_i % 2 == 1:
        print('\nGame {} starts. P1 moves first.'.format(game_i))
    else:
        print('\nGame {} starts. P2 moves first.'.format(game_i))       
    
    # time each game
    time_start = perf_counter()  
    
    # game starts
    # condition to end: finish(win/lose) or turn > board positions (prevent infinity loop)
    while (not game.is_game_over()) and (turn <= board_size**2):
        turn = turn + 1
        # BLUE's turn to move
        if (game_i % 2 == turn % 2): # (odd game AND odd turn) OR (even game AND even turn)
            
            # option - random move for test
            move = random.sample(game.get_allempty(), 1)[0]
            
            # option - minimax move
            #move = Minimax.minimax(board, depth, ntype, p)'['move']'
            
            # option - manual move
            #move_x = input('Insert x-coordinate to move?')
            #move_y = input('Insert y-coordinate to move?')
            #move = (move_x, move_y)
            
            game.place(move, 1)

        # RED's turn to move
        else:
            # option - random move for test
            move = random.sample(game.get_allempty(), 1)[0]
            
            # option - minimax move
            #move = Minimax.minimax(board, depth, ntype, p)['move']
            
            # option - manual move
            #move_x = input('Insert x-coordinate to move?')
            #move_y = input('Insert y-coordinate to move?')
            #move = (move_x, move_y)
        
            game.place(move, 2)
        
        if is_print_midgame:
            print('Game {} - turn {}'.format(game_i, turn))
            game.print()
        
    # stop the count
    time_stop = perf_counter()
    time_elapsed = time_stop - time_start
    
    # print game result
    print('Game {} ends in {} turns. Elapsed time {}.'.format(game_i, turn, time_elapsed))
    if is_print_endgame:
        game.print()
        
    if game.check_win(1):
        print('Game {} won by P1.'.format(game_i))
        winner = p1_id
    elif game.check_win(2):
        print('Game {} won by P2.'.format(game_i))
        winner = p2_id
    else:
        print('NO WINNER! Draw is impossible in Hex, please investigate.')
        winner = None
    
    result_dict = {str(game_i): {'endstate': game,
                                 'winner': winner,
                                 'turns': turn,
                                 'elapsed_time': time_elapsed}}
    games_result['games'].update(result_dict)
    
print('\nAll games completed.')