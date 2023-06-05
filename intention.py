from wimp_surly import *
import bet_game
import learning

from copy import copy

'''
GENERIC INTENTION FUNCTION, TODO
'''
def intends_to_influence(policy, game, variable) -> bool :
    #we intend to influence a variable if : 
    #the expected utility of the policy is <= than that of a reference policy where an intervention has been made doing precisely that influence

    ...

    return False

def intends_to_bring_about(policy, game, outcome) -> bool :

    #we can imagine the outcome is a tuple (variable, outcome) under the hood?
    

    return False


'''
Reference T agent, that acts in response to how S would have acted without S having to actually do it
'''
class ws_T_reference:

    def __init__(self, Q, S):

        #we store in T the actual policy of S, because we need to know how S would have acted
        self.Q = Q

        #we also store the S we're against, because we need to know its type too (probably a way better way of doing this)
        self.S = S

    def policy(self, DS, PSO=False):
        #here, we assume it is the honest policy where S signals honestly its type, and with the added intervention that S can't influence T's type.
        #D_T must be as if controlled by S's regular policy, even though it isn't what S's choose.
        #For that, it seems we need to know what S's would have done

        #We ignore the given DS and compute our own
        return 1 - self.S.policy(self.Q)

    def utility(self, DT, S, X=None):

        if X is not None:
            S.type = X
            
        if S.type == 1:
            return DT

        elif S.type == 0:
            return 1 - DT


class bg_T_interv:

    def __init__(self, Q):
        self.Q = Q # we store Q to know how S would have acted

    def policy(self, DS, x, y, PSO=False):
        if PSO:
            return 0

        #we ignore DS, compute our own.
        #for this, we need to know x,y
        DS = bet_game.S_learner.q_policy(self.Q, x, y)
        #print(f"we return DS={DS} (reference)")
        return DS #our policy here is to trust DS (the one we anticipated)

    def utility(self, DT, X):
        return 1 if DT == X else 0


'''
Hard-Coded ish function testing S's intention to influence T's decision. TODO : better (wimp-surly experiment)
'''
def ws_S_intends_to_influence_D_T(game, Q) -> bool :

    strong = 1
    weak = 0

    #Here we tests whether S intends to make T believe something about its type
    #Precisely, whether it intends to make T believe that its type is strong.

    #The actual policy will be S's trained policy, and T's default nash policy.
    #The reference policy will be one where S was trained specifically in a world where it cannot influence T's belief.

    #Question : How can S's influence on T's belief be tested ? We would need a reference policy where T's belief is already fixed to some value.
    #How to do that? I see how we could easily fix a variable, but a belief? In fact, S doesn't want T to believe anything per se, it is only interested
    #in its decision, which is linked to the abstract notion of its beliefs.

    #Maybe first, I can test whether S intends to influence D_T.

    D_S_strong = wimp_surly.S_learner.q_policy(Q, strong)
    D_T_strong = game.T.policy(D_S_strong)

    D_S_weak = wimp_surly.S_learner.q_policy(Q, weak)
    D_T_weak = game.T.policy(D_S_weak)

    #In that case, the reference policy would be, for S, one where it is trained in a world where D_T is fixed.

    S_ref = wimp_surly.S_learner(game.pure_signalling)
    T_ref = ws_T_reference(Q, S_ref)
    game_ref = wimp_surly(S_ref, T_ref, game.pure_signalling)
    Q_ref = learning.Q_learn(game_ref) #figuring out this matrix might need some refactoring : game should be able to take arbitrary agents, and I should create agents with desirable fixed policies for our test
    
    D_S_strong_ref = wimp_surly.S_learner.q_policy(Q_ref, strong)
    D_T_strong_ref = game.T.policy(D_S_strong_ref)

    D_S_weak_ref = wimp_surly.S_learner.q_policy(Q_ref, weak)
    D_T_weak_ref = game.T.policy(D_S_weak_ref)

    #Here we have only one utility variable, and the policies aren't probabilistic. For the expectation, we need to test for the cases weak and strong

    expected_actual_utility = game.S.p * game.S.utility(D_T_strong, D_S_strong) + (1 - game.S.p) * game.S.utility(D_T_weak, D_S_weak)
    expected_reference_utility = game.S.p * game.S.utility(D_T_strong_ref, D_S_strong_ref) + (1 - game.S.p) * game.S.utility(D_T_weak_ref, D_S_weak_ref)

    #print(f"actual utility : {expected_actual_utility} ; reference utility : {expected_reference_utility}")
    return expected_actual_utility <= expected_reference_utility


def ws_S_intends_to_bring_about_D_T(game, Q, outcome) -> bool :

    intends_to_influence = ws_S_intends_to_influence_D_T(game, Q)

    outcome_possible = True #in our simple hardcoded setting

    #outcome = D_T in this context.

    utility_outcome = game.S.utility(outcome, 1 - outcome) #here we do this because both decisions are always opposite?
    alternative_utility = game.S.utility(1 - outcome, outcome)
    is_best_outcome = utility_outcome >= alternative_utility

    return intends_to_influence and outcome_possible and is_best_outcome


'''
    HARD-CODED FUNCTIONS FOR THE BET-GAME EXPERIMENT
'''

def bg_S_intends_to_influence_D_T(game, Q, Q_ref, PSO=False) -> bool :
    S_ref = game.S
    T_ref = bg_T_interv(Q)
    game_ref = bet_game.bet_game(S_ref, T_ref)
    #TODO : check minimality conditions
    #TODO : learn with PSO when it should be the case

    #important question : should this be shield-trained?
    #we need to know how S would have acted if some variable was influenced for it.
    #but that means the training process should be the same. the only difference should be the fact that the variable is fixed.
    #which means we should train it with the shield as well. which leads to infinite recursion. which is bad.

    #possible way out of this : train the normal agent and the reference agent at the same time?
    #each time we perform a training step for some (x,y), we compute the utility under the normal game, and the utility over the ref game?

    #okay maybe I complicated things too much. what does it mean "if that variable was fixed S wouldn't have done that" in terms of our Q matrix?
    #

    #there's 6 possible settings. for each of them we need to play a normal game and a reference game. that is 12 games.
    actual_utilities = []
    reference_utilities = []
    for y in range(2):
        for x in range(3):
            _, _, _, US, _ = game.play_game(X=x, Y=y, Q=Q, PSO=PSO)
            #S and T have no state so the only difference between the actual and reference games is the Q matrix
            _, _, _, US_ref, _ = game_ref.play_game(X=x, Y=y, Q=Q_ref, is_interv=True, PSO=PSO)

            actual_utilities.append(US)
            reference_utilities.append(US_ref)

    #this is the sum over all settings of the utility for this setting weighted by the proba of the setting 
    expected_actual_utility = expected_utility(actual_utilities, game.p)
    expected_reference_utility = expected_utility(reference_utilities, game.p)

    return expected_actual_utility <= expected_reference_utility

#here we test whether S intends to bring about D_T == outcome, under setting (x, y)
def bg_S_intends_to_bring_about_D_T(game, Q, Q_ref, outcome, x, y, PSO=False) -> bool :
    
    intends_to_influence = bg_S_intends_to_influence_D_T(game, Q, Q_ref, PSO=PSO) # TODO this could be optimized and done only once

    if not PSO:
        outcome_possible = True #in our simple hardcoded setting
    else:
        outcome_possible = outcome==0

    #outcome = D_T in this context.

    utility_outcome = game.S.utility(outcome, x, y)
    alternative_utility_1 = game.S.utility((outcome+1)%3, x, y)
    alternative_utility_2 = game.S.utility((outcome+2)%3, x, y)
    is_best_outcome = utility_outcome >= alternative_utility_1 and utility_outcome >= alternative_utility_2

    return intends_to_influence and outcome_possible and is_best_outcome


def bg_S_intends_to_cause(game, Q, Q_ref, setting, PSO=False) -> bool:

    #it seems in this case X = Y = D_T. the only thing S could want to influence is D_T

    """
        This may build a minimal set of settings for which this holds, TODO 
        1. check for empty set. if this holds, then no set will be minimal, meaning FALSE
        2. check for the set {setting}. if this holds, then return true.
        3. if not, test all combinations, in some sort of increasing order : meaning if we find one, we know it's minimal
            -> bottom up.
        4. return false when we arrived at the whole setting space and it's not valid
    """

    #let's assume we already know Y and W_Y
        # -> in this case Y is always D_T
        # -> W is?

    #1. check that they're subset minimal
        # -> since we have only 1 variable, just check that it doesn't hold for empty set

    #2. compute expected utilities under policy

    #3. compute expected utilities under reference policy

    #4. output comparaison 

    # NOTE : this function will likely do more than asked, so we should keep the results and use them to improve efficency
    


    S_interv = game.S
    T_interv = bg_T_interv(Q)
    game_interv = bet_game.bet_game(S_interv, T_interv)

    actual_utilities = []
    ref_utilities_no_interv = []
    ref_utilities_interv = []

    #First we compute all the utilities we'll need
    for y in range(2):
        for x in range(3):
            _, _, _, US, _ = game.play_game(X=x, Y=y, Q=Q, PSO=PSO)
            #S and T have no state so the only difference between the actual and reference games is the Q matrix
            _, _, _, US_ref_interv, _ = game_interv.play_game(X=x, Y=y, Q=Q_ref, is_interv=True, PSO=PSO)
            _, _, _, US_ref_no_interv, _ = game.play_game(X=x, Y=y, Q=Q_ref, is_interv=False, PSO=PSO)

            actual_utilities.append(US)
            ref_utilities_no_interv.append(US_ref_no_interv)
            ref_utilities_interv.append(US_ref_interv)

    expected_actual_utility = expected_utility(actual_utilities, game.p)

    empty_set = [False for _ in range(6)]
    empty_set_check = check_inequality(expected_actual_utility, ref_utilities_no_interv, ref_utilities_interv, empty_set, game.p)

    if empty_set_check: # in this case, no set containing e can be minimal
        return False
    
    #================ALL=OF=THE=ABOVE=CAN=BE=FACTORED=OUT=NO=DEPENDANCY=ON=SETTING================ TODO

    #now we need to find combinations of runi and rui that satisfy minmality + inequality
    #e = setting has to be part of the set.
    (x,y) = setting
    setting_index = 3*y + x
    setting_set = empty_set
    setting_set[setting_index] = True
    
    minimal_set = find_minimal_set(setting_set, expected_actual_utility, ref_utilities_no_interv, ref_utilities_interv, game.p)
    #if minimal_set is not None:
    #    print(f"setting={setting}, minimal_set={minimal_set}")
    #else:
    #    print(f"minimality not respected!")
    return minimal_set is not None
    
    # other combinations : there's 2^(n-1) - 1 combinations left to check, where n is the cardinality of the setting space.
    # in our case, that's 31 combinations to test. they correspond to the numbers 1-31.
    # idea : we proceed in reverse. the natural thing to do is to do recursively
    # we start with E, and we check the inequality for it. 
    # we then, for e in E, check E - {e} recursively.
    # the problem would be for that that we check the highest cardinality first.
    # but if we start with the empty set and just inverse everything, it works out fine.
    # procedure for finding minimal subset validating inequality : 
    # start with empty set 
    # check inequality
    # if true, return that
    # else, for each element not in S, add it, (set to true), and recursively find a minimal subset


    #true procedure for finding maximal subset validating property in powerset(S): (to remove)
    #start with S
    #check property
    #then if valid return S
    #if not valid, for s in S, recursively check S - {s}. if any returns something, that's a maximal subset.
    



def find_minimal_set(W, e_a_u, r_u_n_i, r_u_i, p): # returns a minimal set validating property, or None if no such set exists.
    if check_inequality(e_a_u, r_u_n_i, r_u_i, W, p):
        return W
    else:
        for i, b in enumerate(W):
            if not b: # an element that could be added to form a potential minimal set
                W_c = copy(W)
                W_c[i] = True
                candidate = find_minimal_set(W_c, e_a_u, r_u_n_i, r_u_i, p)
                if candidate is not None:
                    return candidate
        # if we're here we found no minimal set by looking into supersets of W, so this branch doesn't lead to a minimal set.
        return None


def expected_utility(utilities, p):
    expected_utility = 0
    for i in range(3):
        expected_utility += (1.0 - p) * utilities[i] / 3.0
    for i in range(3, 6):
        expected_utility += p * utilities[i] / 3.0
    return expected_utility

def check_inequality(e_a_u, r_u_n_i, r_u_i, W, p) -> bool:
    # W is a list of booleans, indicating if the i'th element is part of the setting Set.
    r_utilities = []
    for i in range(6):
        r_utilities.append(r_u_i[i] if W[i] else r_u_n_i[i])

    return e_a_u <= expected_utility(r_utilities, p)
            






#======================================TESTS======================================


def test_S_intends_to_influence_D_T(game, Q):

    print("\n\n=============TESTING=S=INTENTION============")

    intends = ws_S_intends_to_influence_D_T(game, Q)
    connector = "does" if intends else "doesn't"
    print(f"S {connector} intend to influence T's decision")

    for outcome in [0, 1]:
        itba = ws_S_intends_to_bring_about_D_T(game, Q, outcome)
        print(f"S intends to bring about D_T={outcome}? {itba}")
