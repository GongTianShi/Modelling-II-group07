import numpy as np

def node_assignment(hubs, cost_matrix):
    cost = cost_matrix[:, hubs]
    assignment = np.argmin(cost, axis=1)
    return hubs[assignment]


def get_cost(cost_matrix, flow_matrix, hub_cost, hubs, col_coef=3, trans_coef=1, dist_coef=2):
    hubs = np.sort(hubs)
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
