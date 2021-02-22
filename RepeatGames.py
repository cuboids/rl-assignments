from search_minimax import minimax
from search_alphabeta import alphabeta
from search_iterativedeepening import iterativedeepening
import random
from time import perf_counter
from HexBoard import Agent
from HexBoard import HexBoard


def play_hex(ngames=1, board_size=11, player1=Agent(name="Alice"), player2=Agent(name="Bob"),
             show_midgame=False, show_endgame=True):
    """
    Script to repeat n games.

    How to use:
    Provide input to set up the game.

    Parameters:
        ngames: (int > 0) the number of games to play
        board_size: (int > 1) the size of the hex board
        player1: (Agent) the first agent
        player2: (Agent) the second agent
        show_midgame: if midgame positions need to be printed.
        show_endgame: if the final position needs to be printed

    Output:
    All game results is stored in dict games_result.
    Data stored: endstate, winner, turns, elapsed_time
    To access individual game e.g. games_result['games']['11']

    Q: How to change each player's decison?
    A: Change the value assigned to 'move' correspondingly.
    There are several options (random move, manual move, minimax).
    But user has to modify corresponding code.
    """
    # set parameters

    print('')
    print(f'{player1.name} (blue) and {player2.name} (red) will play on {board_size}x{board_size}', end=' ')
    print(f'Hex Board for {ngames} round{"" if ngames == 1 else "s"}.')
    print(f'{player1.name} and {player2.name} will move first in odd and even games respectively.')
    print('')

    # initialize games data
    games_result = {'p1': player1, 'p2': player2, 'games': {}}

    for game_i in range(1, ngames + 1):

        game = HexBoard(board_size)  # Initialize board
        assert not game.is_game_over()
        turn = 0  # Reset turn count

        if game_i % 2:
            print('')
            print(f'Game {game_i} starts. {player1.name} moves first.')
        else:
            print('')
            print(f'Game {game_i} starts. {player2.name} moves first.')

        # time each game
        time_start = perf_counter()

        # game starts
        # condition to end: finish(win/lose) or turn > board positions (prevent infinity loop)
        while (not game.is_game_over()) and (turn <= board_size**2):
            turn += 1
            # BLUE's turn to move
            if game_i % 2 == turn % 2:  # (odd game AND odd turn) OR (even game AND even turn)
                if player1.searchby == 'random':
                    print('r')
                    move = random.sample(game.get_allempty(), 1)[0]
                elif player1.searchby == 'minimax':
                    print('m')
                    move = minimax(game, player1.depth, 1)['move']
                elif player1.searchby == 'alphabeta':
                    print('ab')
                    move = alphabeta(game, player1.depth, 1)['move']
                elif player1.searchby == 'alphabetaIDTT':
                    print('idtt')
                    move = iterativedeepening(game, player1.timelimit, 1)['move']
                elif player1.searchby == 'human':
                    move_x = input('[Player 1] Insert x-coordinate to move?')
                    move_y = input('[Player 1] Insert y-coordinate to move?')
                    move = (int(move_x), int(move_y))
                else:
                    print(f'Player type {player1.searchby} not yet supported.')
                    print('Playing a random move.')
                    move = random.sample(game.get_allempty(), 1)[0]
                game.place(move, 1)

            # RED's turn to move
            else:
                if player2.searchby == 'random':
                    move = random.sample(game.get_allempty(), 1)[0]
                elif player2.searchby == 'minimax':
                    move = minimax(game, player2.depth, 2)['move']
                elif player2.searchby == 'alphabeta':
                    move = alphabeta(game, player2.depth, 2)['move']
                elif player2.searchby == 'alphabetaIDTT':
                    move = iterativedeepening(game, player2.timelimit, 2)['move']
                elif player2.searchby == 'human':
                    move_x = input('[Player 2] Insert x-coordinate to move: ')
                    move_y = input('[Player 2] Insert y-coordinate to move: ')
                    move = (int(move_x), int(move_y))
                else:
                    print(f'Player type {player2.searchby} not yet supported.')
                    print('Playing a random move.')
                    move = random.sample(game.get_allempty(), 1)[0]
                game.place(move, 2)

            if show_midgame:
                print('Game {} - turn {}'.format(game_i, turn))
                game.print()

        # stop the count
        time_stop = perf_counter()
        time_elapsed = time_stop - time_start

        # print game result
        print(f'Game {game_i} ends in {turn} turns. Elapsed time {time_elapsed}.')
        if show_endgame:
            print('End state:')
            game.print()

        if game.check_win(1):
            print(f'Game {game_i} won by {player1.name} (blue).')
            winner = player1.name
            player1.rate_1vs1(player2)
        elif game.check_win(2):
            print(f'Game {game_i} won by {player2.name} (red).')
            winner = player2.name
            player2.rate_1vs1(player1)
        else:
            print('NO WINNER! Draw is impossible in Hex, please investigate.')
            winner = None

        result_dict = {str(game_i): {'endstate': game,
                                     'winner': winner,
                                     'turns': turn,
                                     'elapsed_time': time_elapsed}}
        games_result['games'].update(result_dict)

    print('')
    print('All games completed.')
    return games_result


#play_hex(10)