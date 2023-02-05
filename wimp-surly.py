	
import numpy as np
from scipy.stats import bernoulli
import pycid
from belief import *

class S_ws:

	def __init__(self):
		self.p = 0.9
		self.type = bernoulli.rvs(self.p)

	def random_policy(self):
		"""
		"""
		random_action = np.random.choice([0,1])
		return random_action

	def utility(self, DT, DS, X=None):

		if X is not None:
			self.type = X

		us = 0
		if self.type == 0:
			if DS == 1:
				us+= 1
		elif self.type == 1:
			if DS == 0:
				us +=1
		if DT == 1:
			us += 2
		return us 


class T_ws:
	
	def policy(self, DS, PSO=False):
		"""
		We assume T plays the fixed stable Nash policy.
		"""

		if PSO:
			return 1
		else:	
			if DS == 1: 	# if S defends
				return 0  #attack
			elif DS == 0:  # S advances
				return 1  # don't attack

	def utility(self, DT, S, X=None):

		if X is not None:
			S.type = X

		if S.type == 1:
			return DT
		elif S.type == 0:
			return 1 - DT


class wimp_surly:

	def __init__(self):
		self.S = S_ws()
		self.T = T_ws()


	def play_game(self, S, T, init_type=True, PSO=False):
		"""
		Calculates player utilities
		"""
		
		if init_type:
			S.type = bernoulli.rvs(S.p) 	# randomize type
		# DS = S.random_policy()
		# DS = policy(S.type)
		# DS = S.policy
		DS = np.random.choice([0,1])
		DT = T.policy(DS, PSO)
		# print("Actions,", DS, DT)

		UT = T.utility(DT, S)
		US = S.utility(DT, DS)
		return S.type, DS, DT, US, UT

	def Q_learn(self, num_games=100, PSO=False):
		"""
		A basic Q learning agent which learns the utilities for S given actions and type and fixed policy for T
		"""
		Q = np.zeros((2,2))  #innit Q values 2x2 for state x action
		S = S_ws()
		T = T_ws()
		for ep in range(1,num_games):
			t, DS, DT, US, UT = self.play_game(S, T, PSO=PSO)
			# print("Q, ", Q)
			Q[t,DS] = US

		return Q


if __name__ == '__main__':
	
	game = wimp_surly()

	# t, DS, DT, US, UT = game.play_game()
	# print("Type, ", t)	
	# print("Utilities, ",US, UT)

	Q = game.Q_learn(num_games=100, PSO=True)
	print(Q)

	macid = pycid.MACID(
    [("X", "Ds"), ("X", "Ut"), ("X", "Us"), ("Ds", "Dt"), ("Ds", "Us"), ("Dt", "Us"), ("Dt", "Ut")],
    agent_decisions={1: ["Ds"], 2: ["Dt"]},
    agent_utilities={1: ["Us"], 2: ["Ut"]},
	)

	macid.add_cpds(
    Ds=[0,1],
    Dt=[0,1],
    X={0: 0.1, 1: 0.9},
    Us=lambda X, Dt, Ds: game.S.utility(Dt, Ds, X=X),
    Ut=lambda X, Dt: game.T.utility(Dt, game.S, X=X),
    )

	# macid.draw()
	ne = macid.get_ne()
	print(len(ne))
	for i in range(len(ne)):
		print(f"NE {i}")
		print(ne)

	test_S_type_belief(game, Q)
	test_T_type_belief(game)