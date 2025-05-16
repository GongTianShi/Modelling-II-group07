import LoadNetworkData as lnd
import random as rand
import math
from tqdm import tqdm
import numpy as np
from itertools import chain, combinations
import CostHeuristic as ch

def all_subsets(s):
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

# Load network data
small_network_data = lnd.load_data_prefixed(prefix="SMALL", verbose=False)
large_network_data = lnd.load_data_prefixed(prefix="LARGE", verbose=False)

# Cost multipliers
chi = 3  # Collection
alpha = 1  # Transfer
delta = 2  # Distribution

# Cost function to be used by simmulated annealing
def total_cost(n, c, w, f, hub_selections, hub_assignments):
    # # Initialize cost
    # cost = 0

    # # Calculate and add total network costs
    # for i in range(n):
    #     for j in range(n):
    #         h_i = hub_assignments[i]
    #         h_j = hub_assignments[j]
    #         cost += w[i][j] * (chi * c[i][h_i] + alpha * c[h_i][h_j] + delta * c[h_j][j])

    # # Calculate and add total fixed hub costs
    # cost += sum([f[k] for k in hub_selections])

    return ch.get_cost(c, w, f, np.array(list(hub_selections)))

# Makes new hub selection based on the previous
def make_new_hub_selection(n, hub_selections):
    new_hub_selections = set([h for h in hub_selections])

    if len(new_hub_selections) != n:
        add = rand.choice([i for i in range(n) if i not in new_hub_selections])
        new_hub_selections.add(add)
    remove = rand.choice(list(new_hub_selections))
    new_hub_selections.remove(remove)

    if len(new_hub_selections) != 0:
        return new_hub_selections
    else:
        return hub_selections

# Assigns non hub nodes to hubs s.t. the cost from the node to the hub is minimized
def assign_nodes_to_hubs(n, c, hub_selections):
    return [min(hub_selections, key=lambda h: c[i][h]) for i in range(n)]

# Solves the PostNL problem using simulated annealing meta heuristic
def simulated_annealing_solver(network_data, log=True, T0=10e6, Tmin=1, boltzmann_const=1e-28, alpha_decay=0.99, max_iter=10000000):
    # Destructure network data
    n, c, w, f, q, d = network_data

    # Initialize random hub selections and assignments and cost
    current_hub_selections = set(rand.sample(list(range(n)), rand.choice(list(range(1, n+1)))))
    current_hub_assignments = assign_nodes_to_hubs(n, c, current_hub_selections)
    current_total_cost = total_cost(n, c, w, f, current_hub_selections, current_hub_assignments)

    best_hub_selections = current_hub_selections
    best_hub_assignments = current_hub_assignments
    best_total_cost = current_total_cost

    # Initialize temperature
    T = T0

    # Perform Simulated Annealing
    for iteration in range(max_iter):
            if T < Tmin:
                break

            new_hub_selections = make_new_hub_selection(n, current_hub_selections)
            new_hub_assignments = assign_nodes_to_hubs(n, c, new_hub_selections)
            new_total_cost = total_cost(n, c, w, f, new_hub_selections, new_hub_assignments)

            total_cost_diff = new_total_cost - current_total_cost
            if (total_cost_diff < 0) or (rand.random() < math.exp(-total_cost_diff / (boltzmann_const * T))):
                current_hub_selections = new_hub_selections
                current_hub_assignments = new_hub_assignments
                current_total_cost = new_total_cost
                if current_total_cost < best_total_cost:
                    best_hub_selections = current_hub_selections
                    best_hub_assignments = current_hub_assignments
                    best_total_cost = current_total_cost

            T *= alpha_decay  # cool down

    if log:
        print(f"Best hub selections: {best_hub_selections}")
        print(f"Best hub assignments: {best_hub_assignments}")
        print(f"Best total cost: {best_total_cost}")
        print("")
        print(f"Current hub selections: {current_hub_selections}")
        print(f"Current hub assignments: {current_hub_assignments}")
        print(f"Current total cost: {current_total_cost}")

    return best_hub_selections, best_hub_assignments, best_total_cost

print("########## simulated annealing solver solution small network #######################")
simulated_annealing_solver(small_network_data)
print("####################################################################################")
print("")

print("########## simulated annealing solver solution large network #######################")
simulated_annealing_solver(large_network_data)
print("####################################################################################")
print("")

# Brute force approach ##################################################################
def brute_force_solver(network_data, log=True):
    # Destructure network data
    n, c, w, f, q, d = network_data

    lowest_tc = math.inf
    lowest_selection = None
    lowest_assignment = None

    for selection in tqdm(all_subsets(set(list(range(n)))), desc="Brute Force Loop"):
        if len(selection) == 0:
            continue

        assignment = assign_nodes_to_hubs(n, c, selection)

        tc = total_cost(n, c, w, f, selection, assignment)

        if tc < lowest_tc:
            lowest_tc = tc
            lowest_selection = selection
            lowest_assignment = assignment

    if log:
        print(f"Best hub selections: {lowest_selection}")
        print(f"Best hub assignments: {lowest_assignment}")
        print(f"Best total cost: {lowest_tc}")

print("########## brute force solver solution small network #######################")
brute_force_solver(small_network_data)
print("####################################################################################")
print("")

print("########## brute force solver solution large network #######################")
brute_force_solver(large_network_data)
print("####################################################################################")
print("")

#########################################################################################

# Determine optimal constants #########################################################
best_avg_cost_small = math.inf
best_avg_cost_large = math.inf

best_cost_small = None
best_cost_large = None

boltzmann_order_small = None
boltzmann_order_large = None

alpha_decay_small = None
alpha_decay_large = None

best_selection_small = None
best_selection_large = None

best_assignment_small = None
best_assignment_large = None

for alpha_decay in tqdm([0.90, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98], desc="Outer tuning loop"):
    for boltzmann_order in tqdm(range(-50, 50), desc="Inner tuning loop", leave=False):
        boltzmann_order = float(boltzmann_order)

        selections_small = []
        selections_large = []

        assignments_small = []
        assignments_large = []

        costs_small = []
        costs_large = []
        for i in range(10):
            selection_small, assignment_small, cost_small = simulated_annealing_solver(small_network_data, log=False, boltzmann_const=(10.0**boltzmann_order), alpha_decay=alpha_decay)
            selection_large, assignment_large, cost_large = simulated_annealing_solver(large_network_data, log=False, boltzmann_const=(10.0**boltzmann_order), alpha_decay=alpha_decay)

            selections_small.append(selection_small)
            selections_large.append(selection_large)

            assignments_small.append(assignment_small)
            assignments_large.append(assignment_large)

            costs_small.append(cost_small)
            costs_large.append(cost_large)

        avg_cost_small = sum(costs_small)/len(costs_small)
        avg_cost_large = sum(costs_large)/len(costs_large)

        i_min_small = np.argmin(costs_small)
        i_min_large = np.argmin(costs_large)

        cost_small = costs_small[i_min_small]
        selection_small = selections_small[i_min_small]
        assignment_small = assignments_small[i_min_small]

        cost_large = costs_large[i_min_large]
        selection_large = selections_large[i_min_large]
        assignment_large = assignments_large[i_min_large]

        if avg_cost_small <= best_avg_cost_small:
            best_avg_cost_small = avg_cost_small

            boltzmann_order_small = boltzmann_order
            alpha_decay_small = alpha_decay
            best_cost_small = cost_small
            best_selection_small = selection_small
            best_assignment_small = assignment_small

        if avg_cost_large <= best_avg_cost_large:
            best_avg_cost_large = avg_cost_large

            boltzmann_order_large = boltzmann_order
            alpha_decay_large = alpha_decay
            best_cost_large = cost_large
            best_selection_large = selection_large
            best_assignment_large = assignment_large

print("# TUNED PARAMS #######################")
print(f"Small Network => boltzmann_order: {boltzmann_order_small}, alpha_decay: {alpha_decay_small},\n \
    best_avg_cost_small: {best_avg_cost_small}, best_cost_small: {best_cost_small}, selection: {best_selection_small}, assignment: {best_assignment_small}")
print(f"Large Network => boltzmann_order: {boltzmann_order_large}, alpha_decay: {alpha_decay_large},\n \
    best_avg_cost_large: {best_avg_cost_large}, best_cost_large: {best_cost_large}, selection: {best_selection_large}, assignment: {best_assignment_large}")
print("######################################")
#######################################################################################
