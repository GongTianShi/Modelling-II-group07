import LoadNetworkData as lnd
import random as rand
import math

# Load network data
small_network_data = lnd.load_data_prefixed(prefix="SMALL", verbose=False)
large_network_data = lnd.load_data_prefixed(prefix="LARGE", verbose=False)

# Cost multipliers
chi = 3  # Collection
alpha = 1  # Transfer
delta = 2  # Distribution

# Cost function to be used by simmulated annealing
def total_cost(n, c, w, f, hub_selections, hub_assignments):
    # Initialize cost
    cost = 0

    # Calculate and add total network costs
    for i in range(n):
        for j in range(n):
            h_i = hub_assignments[i]
            h_j = hub_assignments[j]
            cost += w[i][j] * (chi * c[i][h_i] + alpha * c[h_i][h_j] + delta * c[h_j][j])

    # Calculate and add total fixed hub costs
    cost += sum([f[k] for k in hub_selections])

    return cost

# Makes new hub selection based on the previous
def make_new_hub_selection(n, hub_selections):
    new_hub_selections = hub_selections

    add = rand.choice([i for i in range(n) if i not in new_hub_selections])
    remove = rand.choice(list(new_hub_selections))
    new_hub_selections.add(add)
    new_hub_selections.remove(remove)

    if len(new_hub_selections) != 0:
        return new_hub_selections
    else:
        return hub_selections

# Assigns non hub nodes to hubs s.t. the cost from the node to the hub is minimized
def assign_nodes_to_hubs(n, c, hub_selections):
    return [min(hub_selections, key=lambda h: c[i][h]) for i in range(n)]

# Solves the PostNL problem using simulated annealing meta heuristic
def simulated_annealing_solver(network_data, T0=1e6, Tmin=1, boltzmann_const=0.1, alpha_decay=0.95, max_iter=1000000):
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
            if total_cost_diff < 0 or rand.random() < math.exp(-total_cost_diff / (boltzmann_const * T)):
                current_hub_selections = new_hub_selections
                current_hub_assignments = new_hub_assignments
                current_total_cost = new_total_cost
                if current_total_cost < best_total_cost:
                    best_hub_selections = current_hub_selections
                    best_hub_assignments = current_hub_assignments
                    best_total_cost = current_total_cost

            T *= alpha_decay  # cool down

    print(f"Best hub selections: {best_hub_selections}")
    print(f"Best hub assignments: {best_hub_assignments}")
    print(f"Best total cost: {best_total_cost}")
    print("")
    print(f"Current hub selections: {current_hub_selections}")
    print(f"Current hub assignments: {current_hub_assignments}")
    print(f"Current total cost: {current_total_cost}")


print("########## simulated annealing solver solution small network #######################")
simulated_annealing_solver(small_network_data)
print("####################################################################")
print("")

print("########## simulated annealing solver solution small network #######################")
simulated_annealing_solver(large_network_data)
print("####################################################################")
print("")
