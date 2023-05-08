import numpy as np
from wimp_surly import *
import deception

def Q_learn(game, num_games=100, PSO=False):
	"""
	A basic Q learning agent which learns the utilities for S given actions and type and fixed policy for T
	"""
	Q = np.zeros((2,2))  #init Q values 2x2 for state x action
	#S = S_learner(game.pure_signalling)
	#T = T_simple_nash()
	for ep in range(1,num_games):
		t, DS, DT, US, UT = game.play_game(PSO=PSO)
		# print("Q, ", Q)
		Q[t,DS] = US

	return Q

def Q_learn_shield(game, num_games=100): # a special kind of shield, where we don't learn bad actions

	Q = np.zeros((2,2))
	S = S_learner(game.pure_signalling)
	T = T_simple_nash()
	for ep in range(1, num_games):

		t, DS, DT, US, UT = game.play_game(S=S, T=T, PSO=False)
		updated_Q = np.array(Q)
		updated_Q[t, DS] = US
		if not deception.S_is_deceptive(game, updated_Q):
			Q = updated_Q
		else:
			#update Q with either 0, or -inf, not sure, to discuss
			Q[t, DS] = -np.inf

	return Q

def Q_learn_reward_shaping(game, num_games=100, punishement=2, set=False): # a special kind of shield, where we don't learn bad actions

	Q = np.zeros((2,2))
	S = S_learner(game.pure_signalling)
	T = T_simple_nash()
	for ep in range(1, num_games):

		t, DS, DT, US, UT = game.play_game(S=S, T=T, PSO=False)
		Q[t, DS] = US
		if deception.S_is_deceptive(game, Q):
			if set:
				Q[t, DS] = -punishement
			else:
				Q[t, DS] -= punishement

	return Q