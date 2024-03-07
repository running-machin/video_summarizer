import numpy as np
from ortools.algorithms.python import knapsack_solver


def knapsack_ortools(values, weights, items, capacity):
    """0-1 Knapsack problem solver"""
    osolver = knapsack_solver.KnapsackSolver(
        knapsack_solver.SolverType.KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER,
        "SummarizationSegmentSelection")

    scale = 1000
    values = np.array(values)
    weights = np.array(weights)
    values = (values * scale).astype(np.int_)
    weights = (weights).astype(np.int_)
    capacity = capacity

    osolver.init(values.tolist(), [weights.tolist()], [capacity])
    osolver.solve()
    packed_items = [x for x in range(0, len(weights))
                    if osolver.best_solution_contains(x)]

    return packed_items
