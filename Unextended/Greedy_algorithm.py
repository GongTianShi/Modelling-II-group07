import numpy as np
from LoadNetworkData import *
from CostHeuristic import *


def greedy_algorithm(cost_matrix, flow_matrix, hub_cost):
    no_nodes = len(hub_cost)
    hubs = np.random.randint(0, no_nodes, 2)#np.array([2, 8])#np.random.randint(0, no_nodes, 2)#np.array([2, 3, 9])#
    maximal_configuration = False
    cost = get_cost(cost_matrix, flow_matrix, hub_cost, hubs)
    totcost = cost
    while not maximal_configuration:
        temp = totcost
        for ii in range(no_nodes):
            if ii not in hubs:
                new_hubs = np.concatenate((hubs, np.array([ii])))
                new_cost = get_cost(cost_matrix, flow_matrix, hub_cost, new_hubs)
                if new_cost < totcost:
                    totcost = new_cost
                    hubs = new_hubs
        for hub in hubs:
            if len(hubs) != 2:
                new_hubs = hubs[hubs != hub]
                new_cost = get_cost(cost_matrix, flow_matrix, hub_cost, new_hubs)
                if new_cost < totcost:
                        totcost = new_cost
                        hubs = new_hubs
        if temp == totcost:
            maximal_configuration = True
    return totcost, hubs







number, cost_mat, flow_mat, hub_cost, orig_flow, dest_flow = load_data_prefixed(prefix="LARGE", verbose=False)
# flow_mat Aij go from i-->j
cost_f, hubs = greedy_algorithm(cost_mat, flow_mat, hub_cost)
print(hubs+1)
print(cost_f)
print(node_assignment(np.sort(hubs), cost_mat)+1)

#print(np.sum(flow_mat, axis = 0))
#print(dest_flow)
