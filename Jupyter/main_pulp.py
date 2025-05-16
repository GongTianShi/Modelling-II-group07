import pulp
import pandas as pd
import numpy as np
import os

# === CONSTANTS ===
THRESHOLD = 1e-6           # minimum flow threshold
BIG_M_DEFAULT = 1e5        # default big M
# Cost multipliers
CHI = 3    # Collection cost multiplier
ALPHA = 1  # Transfer cost multiplier
DELTA = 2  # Distribution cost multiplier

# === 1. Data Loading Function ===
def load_data(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        # Load flow matrix
        w_df = pd.read_excel(file_path, sheet_name="w", index_col=0)
        w = w_df.fillna(0).to_numpy()
        nodes = list(map(int, w_df.index.tolist()))

        # Load cost/distance matrix
        c_df = pd.read_excel(file_path, sheet_name="c", index_col=0)
        c = c_df.fillna(0).to_numpy()

        # Load fixed hub opening costs
        f_df = pd.read_excel(file_path, sheet_name="f", header=None, index_col=0)
        f = f_df.iloc[:, 0].to_dict()
        hubs = list(f.keys())

        # Calculate total outgoing/incoming flows
        q = {node: w[nodes.index(node)].sum() for node in nodes}
        d = {node: w[:, nodes.index(node)].sum() for node in nodes}

        return nodes, hubs, w, c, f, q, d
    except Exception as e:
        raise RuntimeError(f"Error loading data: {e}")

# Prompt for file
file_path = input("Enter Excel file path (.xlsx): ")
nodes, hubs, w, c, f, q, d = load_data(file_path)
print("Data loaded successfully!")

# === 2. Build cost dictionaries ===
C = {(i, k): c[nodes.index(i)][hubs.index(k)] for i in nodes for k in hubs}
G = {(l, j): c[hubs.index(l)][nodes.index(j)] for l in hubs for j in nodes}
W = {(k, l): c[hubs.index(k)][hubs.index(l)] for k in hubs for l in hubs}

# === 3. Define optimization model ===
model = pulp.LpProblem("Hub_and_Spoke", pulp.LpMinimize)

y = pulp.LpVariable.dicts("y", hubs, cat="Binary")
x = pulp.LpVariable.dicts("x", (nodes, hubs), cat="Binary")
t = pulp.LpVariable.dicts("t", (hubs, hubs), lowBound=0)
z = pulp.LpVariable.dicts("z", (hubs, nodes), lowBound=0, upBound=1)

# === 4. Objective ===
model += (
    pulp.lpSum(f[k] * y[k] for k in hubs)
    + CHI * pulp.lpSum(q[i] * C[(i, k)] * x[i][k] for i in nodes for k in hubs)
    + ALPHA * pulp.lpSum(t[k][l] * W[(k, l)] for k in hubs for l in hubs)
    + DELTA * pulp.lpSum(d[j] * G[(l, j)] * z[l][j] for l in hubs for j in nodes)
)

# === 5. Constraints ===
for i in nodes:
    model += pulp.lpSum(x[i][k] for k in hubs) == 1
    for k in hubs:
        model += x[i][k] <= y[k]
for j in nodes:
    model += pulp.lpSum(z[l][j] for l in hubs) == 1
for l in hubs:
    model += (
        pulp.lpSum(q[i] * x[i][l] for i in nodes)
        + pulp.lpSum(t[k][l] for k in hubs)
        == pulp.lpSum(d[j] * z[l][j] for j in nodes)
        + pulp.lpSum(t[l][k] for k in hubs)
    )
big_M = w.sum() if hasattr(w, 'sum') else BIG_M_DEFAULT
for k in hubs:
    for l in hubs:
        model += t[k][l] <= big_M * y[k]
        model += t[k][l] <= big_M * y[l]

# === 6. Solve ===
model.solve()

# === 7. Outputs ===
print(f"\nStatus: {pulp.LpStatus[model.status]}")
print(f"Total Cost: {pulp.value(model.objective):.2f}")

print("\n=== Collection Assignments ===")
for i in nodes:
    for k in hubs:
        if pulp.value(x[i][k]) > THRESHOLD:
            print(f"Node {i} -> Hub {k}: Flow = {q[i] * pulp.value(x[i][k]):.1f}")

# Compute transfer loads based on parcel assignments
transfer_loads = {(k, l): 0.0 for k in hubs for l in hubs}
for i in nodes:
    for j in nodes:
        flow = w[nodes.index(i)][nodes.index(j)]
        if flow > THRESHOLD:
            # find assigned hubs
            k = next((k for k in hubs if pulp.value(x[i][k]) > THRESHOLD), None)
            l = next((l for l in hubs if pulp.value(z[l][j]) > THRESHOLD), None)
            if k is not None and l is not None:
                transfer_loads[(k, l)] += flow

print("\n=== Transfer Assignments ===")
for (k, l), val in transfer_loads.items():
    if val > THRESHOLD:
        print(f"Hub {k} -> Hub {l}: Flow = {val:.1f}")

print("\n=== Distribution Assignments ===")
for j in nodes:
    for l in hubs:
        if pulp.value(z[l][j]) > THRESHOLD:
            print(f"Hub {l} -> Node {j}: Flow = {d[j] * pulp.value(z[l][j]):.1f}")

print("\n=== Full Parcel Flow Paths ===")
for i in nodes:
    for j in nodes:
        flow = w[nodes.index(i)][nodes.index(j)]
        if flow > THRESHOLD:
            k = next((k for k in hubs if pulp.value(x[i][k]) > THRESHOLD), None)
            l = next((l for l in hubs if pulp.value(z[l][j]) > THRESHOLD), None)
            if k and l:
                print(f"Parcel {i} -> {j}: via Hub {k} -> Hub {l} | Amount: {flow:.1f}")
            else:
                print(f"Parcel {i} -> {j}: No valid hub assignment found!")