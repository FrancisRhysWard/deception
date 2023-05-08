import pycid
from wimp_surly import *
from learning import Q_learn, Q_learn_shield, Q_learn_reward_shaping
from belief import *
from intention import test_S_intends_to_influence_D_T
from deception import test_S_deceives_T, S_is_deceptive


if __name__ == '__main__':
	
	pure_signalling = True
	game = wimp_surly(S_learner(pure_signalling), T_simple_nash(), pure_signalling)

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
	test_S_deceives_T(game, shielded_Q)

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
	test_S_deceives_T(game, pso_Q)
