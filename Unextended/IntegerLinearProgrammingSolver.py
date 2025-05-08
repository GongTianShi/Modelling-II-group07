import LoadNetworkData as lnd
import pulp

# Load network data
small_network_data = lnd.load_data_prefixed(prefix="SMALL", verbose=False)
large_network_data = lnd.load_data_prefixed(prefix="LARGE", verbose=False)

# Cost multipliers
chi = 3  # Collection
alpha = 1  # Transfer
delta = 2  # Distribution

# Solve PostNL network using ILP solver
def ILP_network_solver(network_data, verbose=False):
    # Destructure network data
    n, c, w, f, q, d = network_data

    # Initiate the model
    model = pulp.LpProblem("PostNL_Parcel_Optimization", pulp.LpMinimize)

    # Binary variables {0, 1}
    y = [pulp.LpVariable(f"Hub {k+1}", cat="Binary") for k in range(n)] # y_k
    x = [[pulp.LpVariable(f"Collection {i+1} => {k+1}", cat="Binary") for i in range(n)] for k in range(n)] # x_ik
    z = [[pulp.LpVariable(f"Distribution {l+1} => {j+1}", cat="Binary") for l in range(n)] for j in range(n)] # z_lj

    # Continuous variables
    t = [[pulp.LpVariable(f"Transfer {k+1} => {l+1}", lowBound=0) for k in range(n)] for l in range(n)] # t_kl

    # Fixed costs
    fixed_costs = pulp.lpSum(f[k] * y[k] for k in range(n))

    # Collection costs (i -> k)
    collection_costs = pulp.lpSum(chi * c[i][k] * q[i] * x[i][k] for i in range(n) for k in range(n))

    # Transfer costs (k -> l)
    transfer_costs = pulp.lpSum(alpha * c[k][l] * t[k][l] for k in range(n) for l in range(n))

    # Distribution costs (l -> j)
    distribution_costs = pulp.lpSum(delta * c[l][j] * d[j] * z[l][j] for l in range(n) for j in range(n))

    # Objective function
    model += fixed_costs + collection_costs + transfer_costs + distribution_costs

    # Assignment constraints
    for i in range(n):
        model += pulp.lpSum(x[i][k] for k in range(n)) == 1  # Each non-hub assigns to one hub for collection
    for j in range(n):
        model += pulp.lpSum(z[l][j] for l in range(n)) == 1  # Each non-hub assigns to one hub for distribution

    # Hub activation constraints
    for i in range(n):
        for k in range(n):
            model += x[i][k] <= y[k]  # Can only assign to open hubs for collection
    for l in range(n):
        for j in range(n):
            model += z[l][j] <= y[l]  # Can only assign to open hubs for distribution

    # Flow conservation constraints
    for k in range(n):
        model += pulp.lpSum(t[k][l] for l in range(n)) == pulp.lpSum(q[i] * x[i][k] for i in range(n))  # Outflow = Inflow from non-hubs
    for l in range(n):
        model += pulp.lpSum(t[k][l] for k in range(n)) == pulp.lpSum(d[j] * z[l][j] for j in range(n))  # Inflow = Outflow to non-hubs

    model.solve(pulp.PULP_CBC_CMD(msg=verbose))

    print("Status:", pulp.LpStatus[model.status])
    print("Total Cost:", pulp.value(model.objective))

    # Selected hubs
    hubs = [k+1 for k in range(n) if pulp.value(y[k]) == 1]
    print("\nSelected Hubs:", hubs)

    # Collection assignments
    print("\nCollection Assignments:")
    for i in range(n):
        for k in range(n):
            if pulp.value(x[i][k]) == 1:
                print(f"Node {i+1} -> Hub {k+1} (Flow: {q[i]})")

    # Distribution assignments
    print("\nDistribution Assignments:")
    for j in range(n):
        for l in range(n):
            if pulp.value(z[l][j]) == 1:
                print(f"Hub {l+1} -> Node {j+1} (Flow: {d[j]})")

    # Inter-hub transfers
    print("\nInter-Hub Transfers:")
    for k in hubs:
        for l in hubs:
            flow = pulp.value(t[k-1][l-1])
            if flow != 0:
                print(f"Hub {k} -> Hub {l}: {flow:.1f} parcels")

print("########## ILP solver solution small network #######################")
ILP_network_solver(small_network_data, False)
print("####################################################################")
print("")

print("########## ILP solver solution small network #######################")
ILP_network_solver(large_network_data, False)
print("####################################################################")
print("")
