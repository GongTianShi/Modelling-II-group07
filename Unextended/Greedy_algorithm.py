import numpy as np
from LoadNetworkData import *
from numba import njit, prange


def node_assignment(hubs, cost_matrix):
    cost = cost_matrix[:, hubs]
    assignment = np.argmin(cost, axis=1)
    return hubs[assignment]


def get_cost(cost_matrix, flow_matrix, hub_cost, hubs, col_coef=3, trans_coef=1, dist_coef=2):
    no_nodes = len(hub_cost)
    toflow = np.sum(flow_matrix, axis=0)
    fromflow = np.sum(flow_matrix, axis = 1)

    # get node assignment
    node_ass = node_assignment(hubs, cost_matrix)

    # Getting cost to and from
    collection_cost = np.sum(cost_matrix[np.arange(no_nodes), node_ass]*fromflow)*col_coef
    distribution_cost =  np.sum(cost_matrix[np.arange(no_nodes), node_ass]*toflow)*dist_coef

    cost_toandfrom_hub = collection_cost+distribution_cost

    # Getting the cost from hub to hub
    cost = 0
    for hub in hubs:
        flow_to_node = np.sum(flow_matrix[node_ass==hub, :], axis=0)
        cost += np.dot(flow_to_node, cost_matrix[hub, node_ass])

    # Getting the hub costs
    hub_cost = np.sum(hub_cost[hubs])

    return cost*trans_coef+cost_toandfrom_hub+hub_cost


def greedy_algorithm(cost_matrix, flow_matrix, hub_cost):
    no_nodes = len(hub_cost)
    hubs = np.array([2, 3, 9])#np.random.randint(0, no_nodes, 2)#
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


    




number, cost, flow_mat, hub_cost, orig_flow, dest_flow = load_data_prefixed(prefix="SMALL", verbose=False)
# flow_mat Aij go from i-->j
'''hubs = np.array([0, 1])
cost = np.array([[0, 1, 1],
                 [1, 0, 1],
                 [1, 1, 0]])
flow_mat = np.array([[0, 1, 0],
                     [0, 0, 0],
                     [0, 0, 0]])
hub_cost = np.array([1, 1, 1])
print(get_cost(cost, flow_mat, hub_cost, hubs))'''

print(f"costmat {cost}")
print(f"flow_mat {flow_mat}")
print(f"hub_cost{hub_cost}")
cost, hubs = greedy_algorithm(cost, flow_mat, hub_cost)
print(cost)
print(hubs)
#print(np.sum(flow_mat, axis = 0))
#print(dest_flow)





