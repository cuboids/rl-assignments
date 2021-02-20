# Make agents that can play Hex
from trueskill import Rating, rate_1vs1


class Agent:
    def __init__(self, depth=3, eval="random"):
        """Sets up the agent.

        Args:
            depth: an integer representing the eval depth
            eval: a string with the evaluation method.
              Currently supports "random" and "dijkstra"
        """
        self.depth = depth
        self.eval = eval
        self.rating = Rating()

    def rate_1vs1(self, opponent, win):
        """Updates the rating of the agents after playing a 1vs1"""
        rating2 = opponent.rating
        if win:
            self.rating, opponent.rating = rate_1vs1(self.rating, rating2)
        else:
            self.rating, opponent.rating = rate_1vs1(rating2, self.rating)
