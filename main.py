import pycid
import wimp_surly
import bet_game
from learning import general_Q_learn, Q_learn, Q_learn_shield, Q_learn_reward_shaping
from belief import *
from intention import test_S_intends_to_influence_D_T
from deception import ws_test_S_deceives_T, S_is_deceptive, bg_test_S_deceives_T, bg_S_is_deceptive, bg_S_deceives_T

def wimp_surly_experiment(pure_signalling):

	game = wimp_surly.wimp_surly(wimp_surly.S_learner(pure_signalling), wimp_surly.T_simple_nash(), pure_signalling)

	# t, DS, DT, US, UT = game.play_game()
	# print("Type, ", t)	
	# print("Utilities, ",US, UT)

	Q = Q_learn(game, num_games=100, PSO=False)
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
	#ne = macid.get_ne()
	#print(len(ne))
	#for i in range(len(ne)):
	#	print(f"NE {i}")
	#	print(ne)

	test_S_type_belief(game, Q)

	#test_T_type_belief(game)

	#test_S_intends_to_influence_D_T(game, Q)

	#print("\nTESTING=DECEPTION=WITH=LEARNED=POLICY")
	#test_S_deceives_T(game, Q)

	#print("\nTESTING=DECEPTION=WITH=CRAFTED=DECEPTIVE=POLICY")
	#deceptive_Q = np.array([[1,0],[1,0]]) # This policy always attacks, even when S is weak
	#test_S_deceives_T(game, deceptive_Q)

	print("\nTESTING=DECEPTION=WITH=SHIELDED=POLICY")
	shielded_Q = Q_learn_shield(game, num_games=100)
	print(f"shielded Q : {shielded_Q}")
	ws_test_S_deceives_T(game, shielded_Q)

	#print("\nTESTING=DECEPTION=WITH=REWARD=SHAPING")
	#for set in [False, True]:
	#	print(f"when set={set}")
	#	for p in range(-2, 5):
	#		print(f'	when punishement={p}:')
	#		rs_Q = Q_learn_reward_shaping(game, num_games=100, punishement=p, set=set)
	#		print(f"	reward_shaping_Q : {rs_Q}")
	#		print(f"""	-> S {"is" if S_is_deceptive(game, rs_Q) else "isn't"} deceptive""")

	print("\nTESTING=DECEPTION=WITH=PSO=POLICY")
	pso_Q = Q_learn(game, num_games=100, PSO=True)
	print(f"PSO_Q = {pso_Q}")
	ws_test_S_deceives_T(game, pso_Q)

def bet_game_experiment():

	#we use this as a reference policy
	honest_Q = np.array([
		[0., 0., 1.],
		[1., 0., 0.],
		[0., 1., 0.],
		[0., 0., 1.],
		[1., 0., 0.],
		[0., 1., 0.],
	])

	#we use this as a reference policy in the pso case
	impotent_Q = np.array([
		[1., 0., 0.],
		[1., 0., 0.],
		[1., 0., 0.],
		[1., 0., 0.],
		[1., 0., 0.],
		[1., 0., 0.],
	])

	optimal_safe_Q = np.array([
		[2., 0., 0.],
		[0., 2., 0.],
		[0., 0., 2.],
		[1., 0., 0.],
		[0., 1., 0.],
		[0., 0., 1.],
	])

	game = bet_game.bet_game(bet_game.S_learner(), bet_game.T_simple_nash())
	base_Q = general_Q_learn(game, (6,3))

	print(f"Q learned without any mitigation : \n{base_Q}\n")
	print(f"""the unmitigated policy {"is" if bg_S_is_deceptive(game, base_Q, honest_Q) else "isn't"} deceptive\n""")
	#bg_test_S_deceives_T(game, Q)

	pso_Q = general_Q_learn(game, (6,3), PSO=True)

	print(f"Q learned with pso : \n{pso_Q}")
	print(f"""the pso policy {"is" if bg_S_is_deceptive(game, pso_Q, impotent_Q, PSO=True) else "isn't"} deceptive\n""")

	# TODO : a small function that computes expected utility to compare the policies (here pso isn't really that deceptive but it's poor)

	shielded_Q = general_Q_learn(game=game, q_shape=(6,3), deception_check=bg_S_is_deceptive, Q_ref=honest_Q, num_games=250)

	print(f"Q learned with shield : \n{shielded_Q}")
	print(f"""the shielded policy {"is" if bg_S_is_deceptive(game, shielded_Q, honest_Q) else "isn't"} deceptive\n""")


def weird_test():

	game = bet_game.bet_game(bet_game.S_learner(), bet_game.T_simple_nash())

	weird_matrix = np.array(
		[
			[2., 0., 0.,],
			[0., 2., 0.,],
			[0., 0., 2.,],
			[1., 0., 0.,],
			[0., 1., 2.,],
			[0., 0., 1.,]
		]
	)

	#complicated : when x=1, y=1, does S believes that X=2?
	#one should obviously say no, it literally has access to x. it plays 2 only to trick T into playing 2.
	#however, because of the way I built the game, in this case, it will play the same as if X was 2.

	#for x in range(3):
	#	for y in range(2):
	#		deceives = bg_S_deceives_T(game, weird_matrix, (x,y))
	#		print(f"for ({x},{y}) : deceives={deceives}")
	#weird_is_deceptive = bg_S_is_deceptive(game, weird_matrix)
	#print(f"""the weird policy {"is" if weird_is_deceptive else "isn't"} deceptive\n""")

	#apparently the weird policy, when x=1, y=1, we have that S believes that X=Y
	#in that case, S should play 2. which mean it knows X != Y, because it played Y

	s_belief = bg_S_X_belief(game, 1, weird_matrix, 1,1)
	print(f"s_belief = {s_belief}")

	

if __name__ == '__main__':

	#wimp_surly_experiment(False)

	bet_game_experiment()

	#weird_test()
	
