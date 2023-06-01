import numpy as np
import wimp_surly
import bet_game

"""
======================================================
Sort-of generic Belief function, that should be 
applicable to a wide range of belief situations in SCG
======================================================
"""
def believes(policy, counterfactual_policy, policy_input, args_false, args_true) -> bool: #perhaps, we should give the proposition, and derive the setting true and false, and maybe also the counterfactual policy

	D_T = counterfactual_policy(args_true)

	#args false shall be an array of settings where any (or each?) verifies the proposition being false
	responds = True
	for arg in args_false:
		D_F = counterfactual_policy(arg)
		if D_F == D_T:
			responds = False
			break

	acts_as_true = policy(policy_input) == D_T

	return responds and acts_as_true




"""
======================================================
Specific belief tests for the wimp-surly toy example
======================================================
"""

def S_type_belief(game, type, Q) -> bool:
	
	#the setting format here is just the type, in [0, 1], so the policy is a function that takes a type and returns an action

	policy = lambda type : wimp_surly.S_learner.q_policy(Q, type)
	counterfactual_policy = policy #in this special case, S already has access to the proposition

	return believes(policy, counterfactual_policy, game.S.type, [1 - type], type)


def T_type_belief(game, type, D_S):

	#The policy in that case takes S's decision and returns 1 (not attack) or 0 (attack)
	policy = game.T.policy

	#The counterfactual policy also takes the type (and in fact consequently ignores D_S)
	counterfactual_policy = lambda args : args[1]

	args_true = (D_S, type)
	args_false = [(D_S, 1 - type)]

	return believes(policy, counterfactual_policy, D_S, args_false, args_true)



"""
======================================================
Specific belief tests for the bet-game example
======================================================
"""
#we want to see if S is deceptive. that would mean, given X and Y:
#that S intends to make T believe that X=x, even though S knows X != x
#we need to test S belief that X=x, where y corresponds to D_S
#we also need to test T belief that X=x

def bg_S_X_belief(game, x_hyp, Q, actual_x, actual_y) -> bool :

	policy = lambda xy : bet_game.S_learner.q_policy(Q, xy[0], xy[1])
	#in reality, S already knows the value of X and Y, and therefore whether X is equal to any particular x. 
	# so the counterfactual policy is the same.
	# TODO : the counterfactual policy should maybe be another policy trained with one more variable as knowledge
	counterfactual_policy = policy

	#print(f"setting={setting}")

	args_true = [x_hyp, actual_y]
	args_false = [[(x_hyp+i) % 3, actual_y] for i in [1,2]]

	return believes(policy, counterfactual_policy, (actual_x, actual_y), args_false, args_true)


def bg_T_X_belief(game, x, DS) -> bool : 

	policy = game.T.policy

	#T's policy where they know the actual value of X. args = (DS, X)
	counterfactual_policy = lambda args : args[1] # T's optimal policy when it knows X is just to decide X.

	setting = DS

	args_true = [DS, x]
	args_false = [[DS, (x+i) % 3] for i in [1,2]]

	return believes(policy, counterfactual_policy, setting, args_false, args_true)



#=======================================TESTS=======================================

def test_S_type_belief(game, Q):

	def type_name(type):
		return "strong" if type == 1 else "weak"

	print(f"\n\n==========TESTING=S=BELIEFS=========")
	for actual_type in range(0, 2):
		game.S.type = actual_type
		for proposition_type in range(0, 2):

			#new method
			do_believe = S_type_belief(game, proposition_type, Q)
			belief_text = "does believe" if do_believe else "doesnt believe"
			print(f"When S's type is {type_name(actual_type)}, it {belief_text} that it is {type_name(proposition_type)}")

			#print("	-> They yield the same result" if old_do_believe == do_believe else "	-> They differ :(")

def test_T_type_belief(game):

	def type_name(type):
		return "strong" if type == 1 else "weak"

	print(f"\n\n==========TESTING=T=BELIEFS=========")
	for D_S in range(0, 2):
		for proposition_type in range(0, 2):
			do_believe = T_type_belief(game, proposition_type, D_S)
			belief_text = "does believe" if do_believe else "doesnt believe"
			action_text = "defend" if D_S==1 else "attack"
			print(f"When S chooses to {action_text}, T {belief_text} that it is {type_name(proposition_type)}")