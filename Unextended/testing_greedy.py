import numpy as np

def node_assignment(hubs, cost_matrix):
    cost = cost_matrix[:, hubs]
    assignment = np.argmin(cost, axis=1)
    return hubs[assignment]


def get_cost(cost_matrix, flow_matrix, hub_cost, hubs, col_coef=3, trans_coef=1, dist_coef=2):
    no_nodes = len(hub_cost)
    toflow = np.sum(flow_matrix, axis=0)
    fromflow = np.sum(flow_matrix, axis = 1)

    print(f"The flow from all the nodes is {fromflow}")
    print(f"The flow to all the nodes is {toflow}")

    # get node assignment
    node_ass = node_assignment(hubs, cost_matrix)
    print(f"The node assignment is {node_ass}")

    # Getting cost to and from
    collection_cost = np.sum(cost_matrix[np.arange(no_nodes), node_ass]*fromflow)*col_coef
    distribution_cost =  np.sum(cost_matrix[np.arange(no_nodes), node_ass]*toflow)*dist_coef
    print(f"The cost of collecting the packages is {collection_cost}, with vector {cost_matrix[np.arange(no_nodes), node_ass]*fromflow}")
    print(f"The cost of distributing the packages is {distribution_cost}, with vector {cost_matrix[np.arange(no_nodes), node_ass]*toflow}")

    cost_toandfrom_hub = collection_cost+distribution_cost

    # Getting the cost from hub to hub
    cost = 0
    for hub in hubs:
        flow_to_node = np.sum(flow_matrix[node_ass==hub, :], axis=0)
        cost += np.dot(flow_to_node, cost_matrix[hub, node_ass])
        print(f"The packages to hub {hub} is {flow_to_node}, with cost {cost}")

    # Getting the hub costs
    hub_cost = np.sum(hub_cost[hubs])
    print(f"The hub costs are {hub_cost}")
    print(f"The col/dis costs are {cost_toandfrom_hub}")
    print(f"The inter-hub costs are {cost}")

    return cost*trans_coef+cost_toandfrom_hub+hub_cost

cost_matrix = np.array([[0, 1, 2],
                       [1, 0, 3],
                       [2, 3, 0]])

flow_matrix = np.array([[1, 1, 2],
                        [1, 1, 1],
                        [1, 1, 1]])

hub_cost = np.array([10, 10, 10])

print(get_cost(cost_matrix, flow_matrix, hub_cost, np.array([0, 1])))