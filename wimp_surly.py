import numpy as np
from scipy.stats import bernoulli


class S_learner:

	def __init__(self):
		self.p = 0.9
		self.type = bernoulli.rvs(self.p)

	def random_policy(self):
		"""
		"""
		random_action = np.random.choice([0,1])
		return random_action

	def policy(self, Q):
		return S_learner.q_policy(Q, self.type)

	def q_policy(Q, type):
		return np.argmax(Q[type])

	def shielded_policy(Q, type):
		"""
			Here, we'd check that the Q_policy leads us to a safe state,
			and if not use a safe policy instead.
			However, we don't really have a notion of state in this example.
			Also, our notion of safeness depends on the whole policy, not on consequences of individual decisions (I think?)
			So I see no way of implementing a Shield in the way mentioned in Francesco's paper.
		"""
		...

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


class T_simple_nash:
	
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

	def __init__(self, S, T):
		self.S = S
		self.T = T


	def play_game(self, S=None, T=None, init_type=True, PSO=False, Q=None):
		"""
		Calculates player utilities
		"""
		if S is None:
			S=self.S
		if T is None:
			T=self.T
		
		if init_type:
			S.type = bernoulli.rvs(S.p) 	# randomize type
		# DS = S.random_policy()
		# DS = policy(S.type)
		# DS = S.policy
		DS = np.random.choice([0,1]) if Q is None else S.policy(Q)
		DT = T.policy(DS, PSO)
		# print("Actions,", DS, DT)

		UT = T.utility(DT, S)
		US = S.utility(DT, DS)
		return S.type, DS, DT, US, UT