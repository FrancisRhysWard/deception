	
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

	def S_type_belief_old(self, type, Q) -> bool :
		"""
		Determines whether S believes that its type is the given type
		"""

		#First we check that the agent responds to the proposition, i.e, we can make it change it's decision by doing an intervention telling it that it's true or false

		#Then we check that their decision is the one they'd take if they knew it was true.

		#Better road map : compute the agent's decision if they knew it was false : D_F, and true, D_T. Find their actual decision without intervention : D

		#to find those decisions, we should train the agent with the same opponent, and with one more state encoding the value of phi.
		#if we end up evaluating T's belief of S type, it would mean training T with a 3x3 Q matrix, for phi, D_S, and D_T.
		
		# we always take the action with the highest utility given the settings
		# we use the normal policy as counterfactual policy because S already observes its type 
		D_F = np.argmax(Q[1 - type]) #when S's type isn't type
		D_T = np.argmax(Q[type]) #when S's type is type

		responds = D_F != D_T

		#acts_as_true = np.argmax(Q[type]) == np.argmax(Q[self.type]) # that would be a general way to do it
		acts_as_true = type == self.type # but in our narrow case, since S observes type, S acts as if it knew phi is true if and only if phi is true

		#The agent believes if it responds to the proposition with counterfactual knowledge, and it acts as if it's true
		return responds and acts_as_true


	def S_type_belief(self, type, Q) -> bool:
		
		#the setting format here is just the type, in [0, 1], so the policy is a function that takes a type and returns an action

		policy = lambda type: np.argmax(Q[type])
		counterfactual_policy = policy #in this special case, S already has access to the proposition

		setting = self.type
		setting_false = 1 - type
		setting_true = type

		return self.believes(policy, counterfactual_policy, setting, setting_false, setting_true)


	def test_S_type_belief(self, Q):

		def type_name(type):
			return "strong" if type == 1 else "weak"

		print(f"\n\n==========TESTING=S=BELIEFS=========")
		for actual_type in range(0, 2):
			self.type = actual_type
			for proposition_type in range(0, 2):

				##old method
				#old_do_believe = self.S_type_belief_old(proposition_type, Q)
				#belief_text = "does believe" if old_do_believe else "doesnt believe"
				#print(f"[OLD METHOD] When S's type is {type_name(actual_type)}, it {belief_text} that it is {type_name(proposition_type)}")

				#new method
				do_believe = self.S_type_belief(proposition_type, Q)
				belief_text = "does believe" if do_believe else "doesnt believe"
				print(f"When S's type is {type_name(actual_type)}, it {belief_text} that it is {type_name(proposition_type)}")

				#print("	-> They yield the same result" if old_do_believe == do_believe else "	-> They differ :(")


	def T_type_belief(self, type, D_S):

		#The policy in that case takes S's decision and returns 1 (not attack) or 0 (attack)
		policy = self.T_policy

		#The counterfactual policy also takes the type (and in fact consequently ignores D_S)
		counterfactual_policy = lambda settings : settings[1]

		setting = D_S
		setting_true = (D_S, type)
		setting_false = (D_S, 1 - type)

		return self.believes(policy, counterfactual_policy, setting, setting_false, setting_true)

	def test_T_type_belief(self):

		def type_name(type):
			return "strong" if type == 1 else "weak"

		print(f"\n\n==========TESTING=T=BELIEFS=========")
		for D_S in range(0, 2):
			for proposition_type in range(0, 2):
				do_believe = self.T_type_belief(proposition_type, D_S)
				belief_text = "does believe" if do_believe else "doesnt believe"
				action_text = "defend" if D_S==1 else "attack"
				print(f"When S chooses to {action_text}, T {belief_text} that it is {type_name(proposition_type)}")

	def believes(self, policy, counterfactual_policy, setting, setting_false, setting_true) -> bool: #perhaps, we should give the proposition, and derive the setting true and false, and maybe also the counterfactual policy

		D_F = counterfactual_policy(setting_false)
		D_T = counterfactual_policy(setting_true)

		responds = D_F != D_T

		acts_as_true = policy(setting) == D_T

		return responds and acts_as_true


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

	game.test_S_type_belief(Q)
	game.test_T_type_belief()