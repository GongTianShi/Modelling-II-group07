import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from LoadNetworkData_my import load_data

import networkx as nx
import matplotlib.pyplot as plt

torch.manual_seed(23)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'GPU: {torch.cuda.get_device_name(0)} is available')

# Load data
small_excel = 'Neural_combi_opt/SmallNetworkData.xlsx'
large_excel = 'Neural_combi_opt/LargeNetworkData.xlsx'
w, c, f, q, p, nodes = load_data(large_excel)
num_nodes = w.shape[0]

# Parameters
hidden_size = 2048*2
embedding_size = 2048*2
learning_rate = 1e-7
num_epochs = 2000
temperature = 1.5   # small: 1.5  
num_hub_lst = np.arange(4, 5) 

# Convert to tensors and send to device
w_tensor = torch.tensor(w, dtype=torch.float32, device=device)
c_tensor = torch.tensor(c, dtype=torch.float32, device=device)
f_tensor = torch.tensor(f, dtype=torch.float32, device=device).squeeze()
q_tensor = torch.tensor(q, dtype=torch.float32, device=device)
p_tensor = torch.tensor(p, dtype=torch.float32, device=device)

def create_node_features(w, c, f, q, p):
    features = [
        q,
        p,
        f,
        c.sum(1),
        c.sum(0),
        (w * c).sum(1),
        (w * c).sum(0),
        w.sum(1),
        w.sum(0),
    ]
    closeness = 1 / (c.sum(1) + 1e-8)
    betweenness = torch.zeros_like(q)
    features.extend([closeness, betweenness])
    return torch.stack(features, dim=1)

node_features = create_node_features(w_tensor, c_tensor, f_tensor, q_tensor, p_tensor)

class FeatureScaler:
    def __init__(self):
        self.means = None
        self.stds = None
    
    def fit(self, features):
        self.means = features.mean(dim=0, keepdim=True)
        self.stds = features.std(dim=0, keepdim=True)
        self.stds[self.stds < 1e-8] = 1.0
        
    def transform(self, features):
        return (features - self.means) / self.stds

scaler = FeatureScaler()
scaler.fit(node_features)
normalized_node_features = scaler.transform(node_features).to(device)

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        
    def forward(self, x, adj):
        x = self.linear(x)
        x = adj @ x
        x = self.bn(x)
        return F.relu(x)

class GNNEncoder(nn.Module):
    def __init__(self, in_features, hidden_size, out_features, num_layers=6, dropout=0.2):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.layers = nn.ModuleList()
        self.layers.append(GCNLayer(in_features, hidden_size))
        for _ in range(num_layers - 2):
            self.layers.append(GCNLayer(hidden_size, hidden_size))
        self.layers.append(GCNLayer(hidden_size, out_features))
        
    def forward(self, x, adj):
        residual = None
        for i, layer in enumerate(self.layers):
            x_new = layer(x, adj)
            if residual is not None:
                x_new = x_new + residual
            x = F.dropout(x_new, p=self.dropout, training=self.training)
            residual = x_new
        return x

class HubSelector(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.hub_proj = nn.Linear(embedding_size, 1)
        
    def forward(self, embeddings, mask=None):
        scores = self.hub_proj(embeddings).squeeze(-1)
        if mask is not None:
            scores[~mask] = -float('inf')
        return scores

class AttentionAssignmentNetwork(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.query_proj = nn.Linear(embedding_size, embedding_size)
        self.key_proj = nn.Linear(embedding_size, embedding_size)
        self.value_proj = nn.Linear(embedding_size, embedding_size)
        self.out_proj = nn.Linear(embedding_size, 1)

    def forward(self, node_embeddings, hub_indices):
        hub_embeddings = node_embeddings[hub_indices]
        assignments = []
        for i in range(node_embeddings.size(0)):
            if i in hub_indices:
                assignments.append(i)
                continue

            query = self.query_proj(node_embeddings[i])
            keys = self.key_proj(hub_embeddings)
            values = self.value_proj(hub_embeddings)

            scores = torch.matmul(keys, query) / math.sqrt(keys.size(-1))
            attn_weights = F.softmax(scores, dim=0)

            context = torch.matmul(attn_weights.unsqueeze(0), values)
            best_hub = hub_indices[attn_weights.argmax()].item()
            assignments.append(best_hub)

        return torch.tensor(assignments, device=node_embeddings.device)

class FullModel(nn.Module):
    def __init__(self, in_features, hidden_size, embedding_size):
        super().__init__()
        self.encoder = GNNEncoder(in_features, hidden_size, embedding_size)
        self.hub_selector = HubSelector(embedding_size)
        self.assigner = AttentionAssignmentNetwork(embedding_size)

    def forward(self, node_features, adj):
        return self.encoder(node_features, adj)

def normalize_adj(adj):
    rowsum = adj.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

adj_2 = (w_tensor > 0).float()
adj_2 = adj_2 + torch.eye(num_nodes, device=device)
adj = normalize_adj(adj_2).to(device)

def calculate_cost(hubs, assignments, c, w, f, chi=3, alpha=1, delta=2):
    hub_idx = hubs.detach().cpu() if isinstance(hubs, torch.Tensor) else torch.tensor(hubs)
    assign_idx = assignments.detach().cpu() if isinstance(assignments, torch.Tensor) else torch.tensor(assignments)
    c = c.detach().cpu() if isinstance(c, torch.Tensor) else c
    w = w.detach().cpu() if isinstance(w, torch.Tensor) else w
    f = f.detach().cpu() if isinstance(f, torch.Tensor) else f

    total_cost = f[hub_idx].sum().item()
    collection = chi * (c[torch.arange(len(assign_idx)), assign_idx] * w.sum(1)).sum().item()
    k = assign_idx.unsqueeze(1)
    l = assign_idx.unsqueeze(0)
    transfer = alpha * (c[k, l] * w).sum().item()
    distribution = delta * (c[assign_idx, torch.arange(len(assign_idx))] * w.sum(0)).sum().item()
    total = total_cost + collection + transfer + distribution
    return total

def train_step(node_features, adj, model, optimizer, num_hubs, temperature=temperature):
    model.train()
    node_embeddings = model(node_features, adj)
    hub_scores = model.hub_selector(node_embeddings)
    scaled_scores = hub_scores / temperature
    hub_probs = F.softmax(scaled_scores, dim=0)
    hub_indices = torch.multinomial(hub_probs, num_samples=num_hubs, replacement=False)
    assignments = model.assigner(node_embeddings, hub_indices)
    cost = calculate_cost(hub_indices, assignments, c_tensor, w_tensor, f_tensor)
    reward = -(cost / 1e6)
    log_probs = F.log_softmax(hub_scores, dim=0)
    selected_log_probs = log_probs[hub_indices].sum()
    entropy = -(hub_probs * torch.log(hub_probs + 1e-8)).sum()
    loss = -selected_log_probs * reward - 0.05 * entropy
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    return cost, hub_scores.detach(), hub_indices, assignments

def evaluate(model, node_features, adj, num_hubs):
    model.eval()
    with torch.no_grad():
        node_embeddings = model(node_features, adj)
        hub_scores = model.hub_selector(node_embeddings)
        hub_indices = torch.topk(hub_scores, k=num_hubs).indices
        assignments = model.assigner(node_embeddings, hub_indices)
        cost = calculate_cost(hub_indices, assignments, c_tensor, w_tensor, f_tensor)
    return hub_indices.cpu().tolist(), assignments.cpu().tolist(), cost.item()

# Training loop
all_best_solutions = []

for num_hubs in num_hub_lst:
    print(f"\nTraining with {num_hubs} hubs...")
    model = FullModel(normalized_node_features.size(1), hidden_size, embedding_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_cost = float('inf')
    best_hubs = None
    best_assignments = None

    for epoch in range(num_epochs):
        cost_sampled, _, sampled_hubs, sampled_assignments = train_step(
            normalized_node_features, adj, model, optimizer, num_hubs, temperature=temperature
        )

        model.eval()
        with torch.no_grad():
            node_embeddings = model(normalized_node_features, adj)
            hub_scores_greedy = model.hub_selector(node_embeddings)
            greedy_hubs = torch.topk(hub_scores_greedy, k=num_hubs).indices
            greedy_assignments = model.assigner(node_embeddings, greedy_hubs)
            greedy_cost = calculate_cost(greedy_hubs, greedy_assignments, c_tensor, w_tensor, f_tensor)

        if cost_sampled < best_cost:
            best_cost = cost_sampled
            best_hubs = sampled_hubs.cpu().tolist()
            best_assignments = sampled_assignments.cpu().tolist()
        if greedy_cost < best_cost:
            best_cost = greedy_cost
            best_hubs = greedy_hubs.cpu().tolist()
            best_assignments = greedy_assignments.cpu().tolist()

        if (epoch+1) % 100 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}, Sampled Cost: {cost_sampled:.2f}, Greedy Cost: {greedy_cost:.2f}, Best Cost: {best_cost:.2f}")
            print(f"  Current best hubs: {[nodes[h] for h in best_hubs]}")
            print(f"  Current assignments: {[h+1 for h in best_assignments]}")

    all_best_solutions.append({
        "num_hubs": num_hubs,
        "best_cost": best_cost,
        "best_hubs": best_hubs,
        "best_assignments": best_assignments
    })

print("\nSummary of best solutions per number of hubs:")
for sol in all_best_solutions:
    print(f"Hubs: {sol['num_hubs']}, Best Cost: {sol['best_cost']:.2f}")

chosen = min(all_best_solutions, key=lambda x: x['best_cost'])

print(f"\nBest solution for {chosen['num_hubs']} hubs:")
print(f"Selected hubs: {[nodes[h] for h in chosen['best_hubs']]}")
print(f"Assignments: {[h+1 for h in chosen['best_assignments']]}")
print(f"Best cost: {chosen['best_cost']:.2f}")


# paths

import networkx as nx
from mpl_toolkits.mplot3d import Axes3D

print("Nodes array:", nodes)
print("Type of nodes:", type(nodes))
print("Nodes dtype:", nodes.dtype)

# Extract the best solution
best_hubs = chosen['best_hubs']
best_assignments = chosen['best_assignments']

# Convert tensors to numpy for easier manipulation
w = w_tensor.cpu().numpy()
c = c_tensor.cpu().numpy()
f = f_tensor.cpu().numpy()

# Cost multipliers
chi = 3.0
alpha = 1.0
delta = 2.0

# 1. Packages from each node to its assigned hub (collection flows)
print("Packages from each node to its assigned hub:")
node_to_hub_flows = {}
for node_idx in range(num_nodes):
    hub_idx = best_assignments[node_idx]
    node_outflow = w[node_idx, :].sum()  # Total packages from node_idx (o_i = sum_j w_ij)
    if node_outflow > 0:
        print(f"  From node {nodes[node_idx]} to hub {nodes[hub_idx]}: {node_outflow:.2f}")
        node_to_hub_flows[(node_idx, hub_idx)] = node_outflow

# 2. Packages transferred between hubs
print("\nPackages transferred between hubs:")
hub_to_hub_flows = {}
for i in range(num_nodes):
    k = best_assignments[i]  # Hub for origin node i
    for j in range(num_nodes):
        l = best_assignments[j]  # Hub for destination node j
        flow_ij = w[i, j]
        if flow_ij > 0 and k != l:  # Non-zero flows between different hubs
            print(f"  From hub {nodes[k]} to hub {nodes[l]} for flow {nodes[i]}->{nodes[j]}: {flow_ij:.2f}")
            hub_to_hub_flows[(k, l)] = hub_to_hub_flows.get((k, l), 0) + flow_ij

# 3. Packages from each hub to destination nodes (distribution flows)
print("\nPackages from each hub to destination nodes:")
hub_to_node_flows = {}
for hub_idx in best_hubs:
    print(f"\nHub {nodes[hub_idx]} distributes to:")
    for node_idx in range(num_nodes):
        if best_assignments[node_idx] == hub_idx:
            node_inflow = w[:, node_idx].sum()  # Total packages to node_idx (d_j = sum_i w_ij)
            if node_inflow > 0:
                print(f"  To node {nodes[node_idx]}: {node_inflow:.2f}")
                hub_to_node_flows[(hub_idx, node_idx)] = node_inflow


collection_cost = 0.0
for i in range(num_nodes):
    hub_idx = best_assignments[i]
    o_i = w[i, :].sum()
    collection_cost += chi * c[i, hub_idx] * o_i

transfer_cost = 0.0
for i in range(num_nodes):
    k = best_assignments[i]
    for j in range(num_nodes):
        l = best_assignments[j]
        transfer_cost += alpha * c[k, l] * w[i, j]

distribution_cost = 0.0
for j in range(num_nodes):
    hub_idx = best_assignments[j]
    d_j = w[:, j].sum()
    distribution_cost += delta * c[hub_idx, j] * d_j

hub_cost = f[best_hubs].sum()
total_cost = collection_cost + transfer_cost + distribution_cost + hub_cost


G = nx.DiGraph()

# Add nodes for each layer
for i in range(num_nodes):
    node_name = str(nodes[i])  # Convert to string
    G.add_node(f"L1_{node_name}", layer=1, is_hub=(i in best_hubs))  # Origin nodes
    G.add_node(f"L4_{node_name}", layer=4, is_hub=(i in best_hubs))  # Destination nodes
for hub_idx in best_hubs:
    hub_name = str(nodes[hub_idx])  # Convert to string
    G.add_node(f"L2_{hub_name}", layer=2, is_hub=True)  # Hubs for collection
    G.add_node(f"L3_{hub_name}", layer=3, is_hub=True)  # Hubs for transfer

# Add edges
for (node_idx, hub_idx), flow in node_to_hub_flows.items():
    if node_idx != hub_idx:  # Avoid self-loops
        G.add_edge(f"L1_{nodes[node_idx]}", f"L2_{nodes[hub_idx]}", flow=flow, type='collection')
for (hub1_idx, hub2_idx), flow in hub_to_hub_flows.items():
    G.add_edge(f"L2_{nodes[hub1_idx]}", f"L3_{nodes[hub2_idx]}", flow=flow, type='transfer')
for (hub_idx, node_idx), flow in hub_to_node_flows.items():
    if hub_idx != node_idx:  # Avoid self-loops
        G.add_edge(f"L3_{nodes[hub_idx]}", f"L4_{nodes[node_idx]}", flow=flow, type='distribution')

# Set up 3D plot with wider rectangular figure
fig = plt.figure(figsize=(160, 48))  # Much wider with moderate height
ax = fig.add_subplot(111, projection='3d')

# Define node positions (wider xy plane with original z spacing)
pos = {}
num_nodes_layer = max(num_nodes, len(best_hubs))  # Max nodes in any layer
radius = 40.0  # Match wider xy plane
for node in G.nodes(data=True):
    node_id = node[0]
    layer = node[1]['layer']
    name = node_id.split('_')[1]
    try:
        idx = np.where(nodes.astype(str) == name)[0][0]
    except IndexError:
        print(f"Warning: Node name '{name}' not found in nodes array. Using default index.")
        idx = 0
    angle = 2 * np.pi * idx / num_nodes_layer
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    z = layer * 3  # Reverted to original z spacing
    pos[node_id] = (x, y, z)

# Normalize flow for edge width scaling
all_flows = ([flow for flow in node_to_hub_flows.values()] +
             [flow for flow in hub_to_hub_flows.values()] +
             [flow for flow in hub_to_node_flows.values()])
max_flow = max(all_flows) if all_flows else 1.0

# Draw nodes
for node, data in G.nodes(data=True):
    x, y, z = pos[node]
    color = 'red' if data['is_hub'] else 'lightblue'
    size = 400 if data['is_hub'] else 200
    marker = 'o' if data['is_hub'] else 's'  # Circle for hubs, square for non-hubs
    ax.scatter(x, y, z, c=color, s=size, marker=marker, alpha=0.9, edgecolors='black')

# Draw edges with color and width
edge_colors = {'collection': 'blue', 'transfer': 'green', 'distribution': 'purple'}
for u, v, data in G.edges(data=True):
    x = [pos[u][0], pos[v][0]]
    y = [pos[u][1], pos[v][1]]
    z = [pos[u][2], pos[v][2]]
    flow = data['flow']
    edge_type = data['type']
    width = 1 + 4 * (flow / max_flow)  # Scale width by flow
    ax.plot(x, y, z, c=edge_colors[edge_type], linewidth=width, alpha=0.7)
    if flow > 0:
        mid_x = (x[0] + x[1]) / 2
        mid_y = (y[0] + y[1]) / 2
        mid_z = (z[0] + z[1]) / 2 + 0.1  # Slightly offset for visibility
        ax.text(mid_x, mid_y, mid_z, f"{flow:.1f}", size=10, color=edge_colors[edge_type], fontweight='bold')

# Draw node labels
for node, (x, y, z) in pos.items():
    name = node.split('_')[1]
    ax.text(x, y, z + 0.3, name, size=12, color='black', fontweight='bold', ha='center')

# Add layer background planes for context
for z in [3, 6, 9, 12]:  # Reverted to original z spacing
    ax.plot_surface(np.array([[-40, 40], [-40, 40]]),  # Wider xy range -40 to 40
                    np.array([[40, 40], [-40, -40]]),
                    np.array([[z, z], [z, z]]), color='gray', alpha=0.1)

# Add legend for edge types
for edge_type, color in edge_colors.items():
    ax.plot([], [], c=color, label=f"{edge_type.capitalize()} ({edge_type})")
ax.legend(loc='upper left', fontsize=10)

# Set labels, title, and view with updated limits
ax.set_xlabel('X', fontsize=14)
ax.set_ylabel('Y', fontsize=14)
ax.set_zlabel('Layer', fontsize=14)
ax.set_xlim(-40, 40)  # Wider x range
ax.set_ylim(-40, 40)  # Wider y range
ax.set_zticks([3, 6, 9, 12])  # Reverted to original z-ticks
ax.set_zticklabels(['Origin Nodes', 'Node-to-Hub', 'Hub-to-Hub', 'Dest. Nodes'], fontsize=10)
ax.set_title(f"Enhanced 3D Parcel Flow Network\nHubs: {[nodes[h] for h in best_hubs]}", fontsize=14, pad=20)
ax.view_init(elev=20, azim=45)  # Adjust view angle for better perspective

plt.tight_layout()
plt.show()

# Print cost summary for verification
print("\nCost breakdown summary:")
print(f"Collection cost: {collection_cost:.2f}")
print(f"Transfer cost: {transfer_cost:.2f}")
print(f"Distribution cost: {distribution_cost:.2f}")
print(f"Hub fixed cost: {hub_cost:.2f}")
print(f"Total cost: {total_cost:.2f}")
print(f"\nVerification: Best cost from solution: {chosen['best_cost']:.2f}")


