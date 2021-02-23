class TranspositionTable():
    """Contains a dict that uses board state as keys, and stores info about best moves"""
    
    def __init__(self):
        """Constructor, initiate the transposition table"""
        self.table = {}
    
    def is_empty(self):
        """Check if table is empty"""
        return self.table == {}
    
    def store(self, n):
        """Store result node information to TT"""
        key = n['state'].convert_key()
        if key in self.table.keys():  # Board state already exists in TT
            #print('[TT] Found transpositions')
            # Update TT entry
            if n['depth'] >= self.table[key]['depth']:  # Compare search depth
                self.table[key]['depth'] = n['depth']
                self.table[key]['bestmove'] = n['move']
                self.table[key]['score'] = n['score']
                #print('[TT] Updated depth, best move, score in entry')
        else: # Create new TT entry
            value = {'state': n['state'],
                     'depth': n['depth'],
                     'bestmove': n['move'],
                     'score': n['score']}
            key = n['state'].convert_key()
            self.table.update({key: value})
            #print('[TT] Created new entry')
    
    def lookup(self, n, depth):
        """Return look up result from TT"""
        hit = False
        key = n['state'].convert_key()
        if key in self.table.keys():  # Found tranposition in TT
            # Transposition has larger or equal search depth than current search
            if depth <= self.table[key]['depth']: 
                transposition = self.table[key]
                hit = True # Found transposition with useful depth, can return score
                score = transposition['score']
                bestmove = transposition['bestmove']
                return (hit, score, bestmove)
            # Transposition has smaller search depth than current search
            else:
                transposition = self.table[key]
                hit = False
                score = None # Score in TT not useful
                bestmove = transposition['bestmove']  # Return best move to improve move ordering
                return (hit, score, bestmove)
        else: # Transposition not found
            hit = False
            score = None
            bestmove = ()
            return (hit, score, bestmove)
        
    def count_entry(self):
        return len(self.table.keys())