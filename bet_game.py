import numpy as np
from scipy.stats import bernoulli


class S_learner:

	def random_policy(self):
		"""
		"""
		random_action = np.random.choice([0,1,2])
		return random_action

	def policy(self, Q, X, Y):
		return S_learner.q_policy(Q, X, Y)

	def q_policy(Q, X, Y):
		# we have a 6 * 3 matrix. X and Y select the row, on which we do argmax
		row = Y * 3 + X # (0,0) -> 0 ; (2, 1) -> 5 ; good. 
		#print(f"row is {row}, Q is {Q}")
		return np.argmax(Q[row])

	def utility(self, DT, X, Y):
		#when Y is 0, it emulates Y = X. when Y = 1, it emulates Y = (X+1) mod 3
		if Y == 1:
			Y = (X+1) % 3
		else:
			Y = X

		if DT == Y:
			return 2
		elif DT == X:
			return 1
		else:
			return 0


class T_simple_nash:
	
	def policy(self, DS, PSO=False):
		"""
		We assume T plays the fixed stable Nash policy.
		"""

		if PSO: #in pso's case, T doesn't see what S does. it literally has no way of playing the game. to keep it deterministic, we just play 0
			return 0
		else: #the nash policy, where S is either honest or optimal, would always be to believe S
			#print(f"we return DS={DS} (simple_nash)")
			return DS	
			

	def utility(self, DT, X):
		return 1 if DT == X else 0


class bet_game:

	def __init__(self, S, T):
		self.S = S
		self.T = T
		self.X = None
		self.Y = None
		self.p = 0.3


	def play_game(self, S=None, T=None, PSO=False, Q=None, X=None, Y=None, is_interv=False):
		"""
		Calculates player utilities
		"""
		if S is None:
			S=self.S
		if T is None:
			T=self.T
		
		self.X = X if X is not None else np.random.choice([0,1,2])	# randomize type
		self.Y = Y if Y is not None else np.random.rand() < self.p

		DS = np.random.choice([0,1,2]) if Q is None else S.policy(Q, self.X, self.Y)
		DT = T.policy(DS, self.X, self.Y, PSO=PSO) if is_interv else T.policy(DS, PSO)
		# print("Actions,", DS, DT)

		UT = T.utility(DT, self.X)
		US = S.utility(DT, self.X, self.Y)

		return self.Y * 3 + self.X, DS, DT, US, UT