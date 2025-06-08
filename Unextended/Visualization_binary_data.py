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
cost_states[0] = np.mean(cost_states)
distance = distance + distance.T

print(f"Starting the MDS calculations")
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
mds_coords = mds.fit_transform(distance)
print(mds_coords[0, :])


print(f"Starting the plotting")
xi = np.linspace(min(mds_coords[:, 0]), max(mds_coords[:, 0]), 1000)
yi = np.linspace(min(mds_coords[:, 1]), max(mds_coords[:, 1]), 1000)
xi, yi = np.meshgrid(xi, yi)
zi = griddata((mds_coords[:, 0], mds_coords[:, 1]), cost_states, (xi, yi), method='nearest')

#Making the heatmap
special_comb = np.array([0,0,0,1,0,0,0,0,0,1])
special_idx = np.where((combinations == special_comb).all(axis=1))[0][0]
special_x, special_y = mds_coords[special_idx, 0], mds_coords[special_idx, 1]
xi_flat = xi[0, :]
yi_flat = yi[:, 0]
x_img_idx = np.argmin(np.abs(xi_flat - special_x))
y_img_idx = np.argmin(np.abs(yi_flat - special_y))

plt.figure(figsize=(10, 8))
img = plt.imshow(zi, extent=(xi.min(), xi.max(), yi.min(), yi.max()), origin='lower', aspect='auto')
cbar = plt.colorbar(img)
cbar.set_label('Cost')
plt.xlabel('MDS Dimension 1')
plt.ylabel('MDS Dimension 2')
plt.scatter(special_x, special_y, color='red', marker='*', s=100, label='{4, 10}')
plt.legend()
plt.show()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(xi, yi, zi, cmap='viridis', edgecolor="none")

ax.set_xlabel("MDS Dimension 1")
ax.set_ylabel("MDS Dimension 2")
ax.set_zlabel("Cost")
ax.set_title("The cost landscape of the small network")

plt.show()