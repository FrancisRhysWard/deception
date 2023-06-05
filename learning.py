import numpy as np
from wimp_surly import *
import deception

def general_Q_learn(game, q_shape, num_games=100, deception_check=None, Q_ref=None, pnsh=None, **game_args):
	if deception_check is not None:
		assert Q_ref is not None

	Q = np.array(Q_ref) if Q_ref is not None else np.zeros(q_shape)
	for ep in range(1, num_games):
		idx, DS, DT, US, UT = game.play_game(**game_args)
		u_Q = np.array(Q)
		u_Q[idx, DS] = US
		h = False
		#if idx==5 and DS==1:
			#print(f"(5,1) -> US={US}, and Q=\n{Q}")
			#h = True
		if deception_check is None or not deception_check(game, u_Q, Q_ref):
			Q = u_Q
			Q_ref = Q
			#if h:
				#print(f"and we update")
		else:
			if pnsh is not None:
				Q[idx, DS] = pnsh
			#if h:
				#print(f"but we don't update")

	return Q



def Q_learn(game, num_games=100, PSO=False):
	return general_Q_learn(game, (2,2), num_games=num_games, PSO=PSO)

def Q_learn_shield(game, num_games=100): # a special kind of shield, where we don't learn bad actions
	return general_Q_learn(game, (2,2), num_games=num_games, PSO=False, shielding=True, pnsh=-np.inf)

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