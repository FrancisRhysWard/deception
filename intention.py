from wimp_surly import *
import bet_game
import learning

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


class bg_T_reference:

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

def expected_utility(utilities, p):
    expected_utility = 0
    for i in range(3):
        expected_utility += (1.0 - p) * utilities[i] / 3.0
    for i in range(3, 6):
        expected_utility += p * utilities[i] / 3.0
    return expected_utility


def bg_S_intends_to_influence_D_T(game, Q, Q_ref, PSO=False) -> bool :
    S_ref = game.S
    T_ref = bg_T_reference(Q)
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
    #Q_ref = learning.general_Q_learn(game=game_ref, q_shape=(6,3), is_ref=True) # TODO : factor that out. we probably don't need to recompute it for every setting.

    #there's 6 possible settings. for each of them we need to play a normal game and a reference game. that is 12 games.
    actual_utilities = []
    reference_utilities = []
    for y in range(2):
        for x in range(3):
            _, _, _, US, _ = game.play_game(X=x, Y=y, Q=Q, PSO=PSO)
            #S and T have no state so the only difference between the actual and reference games is the Q matrix
            _, _, _, US_ref, _ = game_ref.play_game(X=x, Y=y, Q=Q_ref, is_ref=True, PSO=PSO)

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






#======================================TESTS======================================


def test_S_intends_to_influence_D_T(game, Q):

    print("\n\n=============TESTING=S=INTENTION============")

    intends = ws_S_intends_to_influence_D_T(game, Q)
    connector = "does" if intends else "doesn't"
    print(f"S {connector} intend to influence T's decision")

    for outcome in [0, 1]:
        itba = ws_S_intends_to_bring_about_D_T(game, Q, outcome)
        print(f"S intends to bring about D_T={outcome}? {itba}")
