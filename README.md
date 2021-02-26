# Reinforcment Learning: Hex
Reinforcement Learning agents for Hex

Dependencies: names (use `!pip install names`), TrueSkill, NumPy, and Matplotlib.

To make new players, use the following syntax:

```
Alice = Agent(name="Alice", searchby="alphabeta")
Bob = Human(name="Bob")
```

The parameter "searchby" indicates the search strategy for the agent. Possible values include "random", "minimax", "alphabeta", "alphabetaIDTT", and "mcts".

To let players play a match, use play_hex:

```
play_hex(ngames=2, Alice, Bob, board_size=6)
Alice.plot_rating_history(Bob).  # Make a plot comparing Alice's and Bob's ratings
```

