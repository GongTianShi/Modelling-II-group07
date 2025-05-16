import numpy as np
import itertools
from numba import njit, prange
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from LoadNetworkData import *
from CostHeuristic import *
from scipy.interpolate import griddata


@njit
def hamming_distance(statea, stateb):
    return np.sum(np.abs(statea-stateb))

def combination_to_hubs(comb):
    return np.where(comb==1)[0]

number, cost, flow_mat, hub_cost, orig_flow, dest_flow = load_data_prefixed(prefix="SMALL", verbose=False)

entries = [0, 1]
length = number
combinations = np.array(list(itertools.product(entries, repeat=length)))
distance = np.zeros((np.shape(combinations)[0], np.shape(combinations)[0]))
cost_states = np.zeros(np.shape(combinations)[0])

print(f"Starting the combination calculations")
for ii in prange(np.shape(combinations)[0]):
    if ii == 0:
        cost_states[ii] = 0
    else:
        cost_states[ii] = get_cost(cost, flow_mat, hub_cost, combination_to_hubs(combinations[ii, :]))
    for jj in range(ii+1):
        distance[ii, jj] = hamming_distance(combinations[ii, :], combinations[jj, :])
distance = distance + distance.T

print(f"Starting the MDS calculations")
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
mds_coords = mds.fit_transform(distance)


print(f"Starting ")
xi = np.linspace(min(mds_coords[:, 0]), max(mds_coords[:, 0]), 100)
yi = np.linspace(min(mds_coords[:, 1]), max(mds_coords[:, 1]), 100)
xi, yi = np.meshgrid(xi, yi)

zi = griddata((mds_coords[:, 0], mds_coords[:, 1]), cost_states, (xi, yi), method='cubic')


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(xi, yi, zi, cmap='viridis', edgecolor="none")

ax.set_xlabel("MDS Dimension 1")
ax.set_ylabel("MDS Dimension 2")
ax.set_zlabel("Value")
ax.set_title("3D MDS Plot")

plt.show()