from time import perf_counter
from HexBoard import Agent, HexBoard


def play_hex(ngames=None, player1=Agent(name="Alice"), player2=Agent(name="Bob"), board_size=3,
             show_midgame=False, show_endgame=True, seed=0, analysis=False):
    # Docstring needs to be revised!
    """
    Script to repeat n games.

    How to use:
    Provide input to set up the game.

    Arguments:
        ngames: (int > 0) the number of games to play
        board_size: (int > 1) the size of the hex board
        player1: (Agent) the first agent
        player2: (Agent) the second agent
        show_midgame: if midgame positions need to be printed.
        show_endgame: if the final position needs to be printed
        seed: specify to get different results
        analysis: if agents should share their thought processes
    Returns:
        All game results is stored in dict games_result.
        Data stored: endstate, winner, turns, elapsed_time
        To access individual game e.g. games_result['games']['11']

    Q: How to change each player's decison?
    A: Change the value assigned to 'move' correspondingly.
    There are several options (random move, manual move, minimax).
    But user has to modify corresponding code.
    """

    if ngames is None:
        ngames = int(input("Number of games: "))

    print(f'{player1.name} (blue) and {player2.name} (red) will play on {board_size}x{board_size}', end=' ')
    print(f'Hex board for {ngames} round{"" if ngames == 1 else "s"}.')
    print(f'{player1.name} and {player2.name} will move first in odd and even games respectively.')
    print()

    # Initialize games data
    games_result = {'p1': player1, 'p2': player2, 'games': {}}
    player1.color, player2.color = HexBoard.BLUE, HexBoard.RED  # Let agents know their colors
    player1.seed = player2.seed = seed
    players = (player1, player2)  # This will help us when players alternate turns

    for game_i in range(1, ngames + 1):
        game = HexBoard(board_size)
        player1.game = player2.game = game_i
        nodes1, nodes2 = [], []
        nodes = (nodes1, nodes2)
        n_turns = 0
        print(f'Game {game_i} starts. {player1.name if game_i % 2 else player2.name} moves first.')
        time_start = perf_counter()  # Time each game

        # Game starts
        # Condition to end: finish(win/lose) or n_turns > board positions (prevent infinity loop)
        while not game.is_game_over() and n_turns < board_size ** 2:

            n_turns += 1
            turn = int(n_turns % 2 != game_i % 2)
            player1.n_turns = player2.n_turns = n_turns
            if analysis:
                nodes[turn].append(players[turn].analyse_position(game))
            move = players[turn].make_move(game)
            game.place(move, players[turn].color)

            if show_midgame:
                print(f'Game {game_i} - Number of moves {n_turns}')
                game.print()

        time_stop = perf_counter()  # Stop the count
        time_elapsed = time_stop - time_start

        # print game result
        print(f'Game {game_i} ends in {n_turns} turns. Elapsed time {time_elapsed}.')
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

        result_dict = {str(game_i): {'endstate': game, 'winner': winner, 'turns': n_turns,
                                     'elapsed_time': time_elapsed, 'nodes': nodes}}
        games_result['games'].update(result_dict)

    print()
    print('All games completed.')
    return games_result


example = True
if example:
    Alice = Agent(name='Alice', searchby='alphabetaIDTT')
    Bob = Agent(name='Bob', searchby="alphabeta")
    result = play_hex(1, Alice, Bob, 5, analysis=True)
    print(result['games']['1']['nodes'])
