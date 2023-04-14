from wimp_surly import *
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
class T_reference:

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
        return self.S.policy(Q)

    def utility(self, DT, S, X=None):

        if X is not None:
            S.type = X
            
        if S.type == 1:
            return DT

        elif S.type == 0:
            return 1 - DT


'''
Hard-Coded ish function testing S's intention to influence T's decision. TODO : better
'''
def S_intends_to_influence_D_T(game, Q) -> bool :

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

    D_S_strong = S_learner.q_policy(Q, strong)
    D_T_strong = game.T.policy(D_S_strong)

    D_S_weak = S_learner.q_policy(Q, weak)
    D_T_weak = game.T.policy(D_S_weak)

    #In that case, the reference policy would be, for S, one where it is trained in a world where D_T is fixed.

    S_ref = S_learner()
    T_ref = T_reference(Q, S_ref)
    game_ref = wimp_surly(S_ref, T_ref)
    Q_ref = learning.Q_learn(game_ref) #figuring out this matrix might need some refactoring : game should be able to take arbitrary agents, and I should create agents with desirable fixed policies for our test
    
    D_S_strong_ref = S_learner.q_policy(Q_ref, strong)
    D_T_strong_ref = game.T.policy(D_S_strong_ref)

    D_S_weak_ref = S_learner.q_policy(Q_ref, weak)
    D_T_weak_ref = game.T.policy(D_S_weak_ref)

    #Here we have only one utility variable, and the policies aren't probabilistic. For the expectation, we need to test for the cases weak and strong

    expected_actual_utility = game.S.p * game.S.utility(D_T_strong, D_S_strong) + (1 - game.S.p) * game.S.utility(D_T_weak, D_S_weak)
    expected_reference_utility = game.S.p * game.S.utility(D_T_strong_ref, D_S_strong_ref) + (1 - game.S.p) * game.S.utility(D_T_weak_ref, D_S_weak_ref)

    #print(f"actual utility : {expected_actual_utility} ; reference utility : {expected_reference_utility}")
    return expected_actual_utility <= expected_reference_utility


def S_intends_to_bring_about_D_T(game, Q, outcome) -> bool :

    intends_to_influence = S_intends_to_influence_D_T(game, Q)

    outcome_possible = True #in our simple hardcoded setting

    #outcome = D_T in this context.

    utility_outcome = game.S.utility(outcome, 1 - outcome) #here we do this because both decisions are always opposite?
    alternative_utility = game.S.utility(1 - outcome, outcome)
    is_best_outcome = utility_outcome >= alternative_utility

    return intends_to_influence and outcome_possible and is_best_outcome


def test_S_intends_to_influence_D_T(game, Q):

    print("\n\n=============TESTING=S=INTENTION============")

    intends = S_intends_to_influence_D_T(game, Q)
    connector = "does" if intends else "doesn't"
    print(f"S {connector} intend to influence T's decision")

    for outcome in [0, 1]:
        itba = S_intends_to_bring_about_D_T(game, Q, outcome)
        print(f"S intends to bring about D_T={outcome}? {itba}")
