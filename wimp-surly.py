	
import numpy as np
from scipy.stats import bernoulli
import pycid

class diplomacy_toy:
	"""
	S's type  = 1 (strong) or 0 (weak);
	DS = 1 (defend) or 0 (attack);
	DT = 1 (not attack) or 0 (attack)
	"""

	def __init__(self):

		self.p = 0.9 	# probability of type 1  / strong
		self.type = bernoulli.rvs(self.p) 	# type 1 or 0
		self.policy = np.random.choice([0,1])

	def T_policy(self, DS, PSO=False):
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


	def random_policy(self):
		"""
		
		"""
		random_action = np.random.choice([0,1])
		return random_action

	def UT(self, DT, X=None):

		if X is not None:
			self.type = X

		if self.type ==1:
			return DT
		elif self.type ==0:
			return 1 - DT

	def US(self, DT, DS, X=None):

		if X is not None:
			self.type = X

		us = 0
		if self.type == 0:
			if DS == 1:
				us+= 1
		elif self.type ==1:
			if DS == 0:
				us +=1
		if DT == 1:
			us += 2
		return us 

	def play_game(self, init_type=True, PSO=False):
		"""
		Calculates player utilities
		"""
		
		if init_type:
			self.type = bernoulli.rvs(self.p) 	# randomize type
		# DS = self.random_policy()
		# DS = policy(self.type)
		# DS = self.policy
		DS = np.random.choice([0,1])
		DT = self.T_policy(DS, PSO)
		# print("Actions,", DS, DT)

		UT = self.UT(DT)
		US = self.US(DT, DS)
		return self.type, DS, DT, US, UT


	def Q_learn(self, num_games=100, PSO=False):
		"""
		A basic Q learning agent which learns the utilities for S given actions and type and fixed policy for T
		"""
		Q = np.zeros((2,2))  #innit Q values 2x2 for state x action
		for ep in range(1,num_games):
			t, DS, DT, US, UT = self.play_game(PSO=PSO)
			# print("Q, ", Q)
			Q[t,DS] = US

		return Q


if __name__ == '__main__':
	
	game = diplomacy_toy()

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
    Us=lambda X, Dt, Ds: game.US(Dt, Ds, X=X),
    Ut=lambda X, Dt: game.UT(Dt, X=X),
    )

	# macid.draw()
	ne = macid.get_ne()
	print(len(ne))
	for i in range(len(ne)):
		print(f"NE {i}")
		print(ne)