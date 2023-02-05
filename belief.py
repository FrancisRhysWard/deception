import numpy as np


"""
======================================================
Sort-of generic Belief function, that should be 
applicable to a wide range of belief situations in SCG
======================================================
"""
def believes(policy, counterfactual_policy, setting, setting_false, setting_true) -> bool: #perhaps, we should give the proposition, and derive the setting true and false, and maybe also the counterfactual policy

	D_F = counterfactual_policy(setting_false)
	D_T = counterfactual_policy(setting_true)

	responds = D_F != D_T

	acts_as_true = policy(setting) == D_T

	return responds and acts_as_true




"""
======================================================
Specific belief tests for the wimp-surly toy example
======================================================
"""

def S_type_belief(game, type, Q) -> bool:
	
	#the setting format here is just the type, in [0, 1], so the policy is a function that takes a type and returns an action

	policy = lambda type: np.argmax(Q[type])
	counterfactual_policy = policy #in this special case, S already has access to the proposition

	setting = game.S.type
	setting_false = 1 - type
	setting_true = type

	return believes(policy, counterfactual_policy, setting, setting_false, setting_true)


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


def T_type_belief(game, type, D_S):

	#The policy in that case takes S's decision and returns 1 (not attack) or 0 (attack)
	policy = game.T.policy

	#The counterfactual policy also takes the type (and in fact consequently ignores D_S)
	counterfactual_policy = lambda settings : settings[1]

	setting = D_S
	setting_true = (D_S, type)
	setting_false = (D_S, 1 - type)

	return believes(policy, counterfactual_policy, setting, setting_false, setting_true)

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