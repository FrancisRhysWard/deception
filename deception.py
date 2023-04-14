from intention import intends_to_bring_about, S_intends_to_bring_about_D_T
from belief import S_type_belief, T_type_belief
from wimp_surly import wimp_surly


'''
Generic Deception Function, TODO
'''
def deception(policy, ref_policy, game, ) -> bool :

    # Deception is to intentionally cause to have a false belief 
    # that is not believed to be true

    # 1) S intends to bring about D_T
    # 2) T believes proposition and proposition is false
    # 3) S doesn't believe phi

    outcome = (..., ...) # (sth that designates D_T, the actual outcome of D_T)
    S_intends = intends_to_bring_about(policy, game, outcome)

    T_believes = believes(policy, counterfactual_policy, setting, args_false, args_true)
    S_believes = ...

    return S_intends and T_believes and not S_believes and phi.value


def S_deceives_T(game: wimp_surly, Q, setting) -> bool:
    #Tests whether S deceives T about its type under given setting (=S.type)
    #the proposition phi here would be that the type is the opposite of what it actually is.
    
    #We know that T's belief of S.type is a bijection of D_T, so we can reason with D_T to know about S.type

    game.S.type = setting

    _, D_S, D_T, U_S, U_T = game.play_game(init_type=False, Q=Q)
    S_intends = S_intends_to_bring_about_D_T(game, Q, outcome=D_T) #equivalent to S's intends to bring about T's belief about the type

    T_believes = T_type_belief(game, 1 - game.S.type, D_S) #whether T is mistaken about S's type
    S_believes = S_type_belief(game, 1 - game.S.type, Q) #whether S is mistaken about its own type

    prop_value = False # hard coded. the type isn't equal to its opposite

    return S_intends and T_believes and (not prop_value) and (not S_believes)

def S_is_deceptive(game: wimp_surly, Q) -> bool:
    for setting in range(2):
        if S_deceives_T(game, Q, setting):
            return True
    return False


def test_S_deceives_T(game: wimp_surly, Q) -> bool:

    for setting in range(2):
        deceives = S_deceives_T(game, Q, setting)
        print(f"When S's type is {setting}, S " + ("is" if deceives else "isn't") + " deceptive")