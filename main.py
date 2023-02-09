import pycid
from wimp_surly import *
from learning import Q_learn
from belief import *
from intention import test_S_intends_T_belief


if __name__ == '__main__':
	
	game = wimp_surly(S_learner(), T_simple_nash())

	# t, DS, DT, US, UT = game.play_game()
	# print("Type, ", t)	
	# print("Utilities, ",US, UT)

	Q = Q_learn(game, num_games=100, PSO=True)
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

	test_S_intends_T_belief(game, Q)