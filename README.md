# Reinforcment Learning: Hex
Reinforcement Learning agents for Hex

Dependencies: names (use `!pip install names`), TrueSkill, NumPy, and Matplotlib.

To make new players, use the following syntax (players can be humans or agents):

```
Alice = Agent(name="Alice", searchby="alphabeta")
Bob = Human(name="Bob")
```

The parameter "searchby" indicates the search strategy for the agent. Possible values include "random", "minimax", "alphabeta", "alphabetaIDTT", and "mcts".

To let players play a match, use play_hex. Here's an example:

```
play_hex(ngames=2, Alice, Bob, board_size=6). # Play two games on a 6x6 hex board.
Alice.plot_rating_history(Bob).  # Make a plot comparing Alice's and Bob's ratings
```

