import numpy as np
from wimp_surly import *

def Q_learn(game, num_games=100, PSO=False):
	"""
	A basic Q learning agent which learns the utilities for S given actions and type and fixed policy for T
	"""
	Q = np.zeros((2,2))  #innit Q values 2x2 for state x action
	S = S_learner()
	T = T_simple_nash()
	for ep in range(1,num_games):
		t, DS, DT, US, UT = game.play_game(S, T, PSO=PSO)
		# print("Q, ", Q)
		Q[t,DS] = US

	return Q