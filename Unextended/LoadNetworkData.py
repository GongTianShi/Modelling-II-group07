import pandas as pd
import numpy as np
import os

CURRENT_DIRPATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIRPATH = os.path.join(CURRENT_DIRPATH, "data")
NETWORK_DATA_SMALL_FILEPATH = os.path.join(DATA_DIRPATH, "SmallNetworkData.xlsx")
NETWORK_DATA_LARGE_FILEPATH = os.path.join(DATA_DIRPATH, "LargeNetworkData.xlsx")

def load_data(file_path=NETWORK_DATA_SMALL_FILEPATH):
    try:
        # Load flow matrix (w)
        w_df = pd.read_excel(file_path, sheet_name="w", header=0, index_col=0)
        w = w_df.fillna(0).to_numpy()
        w = np.array(w)

        # Get nodes - ensure they're consecutive integers starting from 1
        n = len(list(map(int, w_df.index.tolist())))

        # Load distance matrix (c)
        c_df = pd.read_excel(file_path, sheet_name="c", header=0, index_col=0)
        c = c_df.fillna(0).to_numpy()
        c = np.array(c)

        # Load fixed costs (f)
        f_df = pd.read_excel(file_path, sheet_name="f", header=None, index_col=0)
        f = np.array(f_df.iloc[:, 0].to_list())

        # Calculate total flow originating at node i (q_i)
        q = np.array([sum(w[i][j] for j in range(n)) for i in range(n)])

        # Calculate total flow with destination node i (d_i)
        d = np.array([sum(w[j][i] for j in range(n)) for i in range(n)])

        return n, c, w, f, q, d

    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None, None, None

# Prefix options: SMALL, LARGE
def load_data_prefixed(prefix="SMALL", verbose=False):
    n, c, w, f, q, d = load_data({
        "SMALL": NETWORK_DATA_SMALL_FILEPATH,
        "LARGE": NETWORK_DATA_LARGE_FILEPATH
    }[prefix.upper()])

    if n == 0:
        print(f"Failed to load data {prefix.lower()} network. Please check the file path and format.")
    else:
        print(f"{prefix.capitalize()} network data loaded successfully:")
        if verbose: print(f"Distance Matrix (c):\n {c}")
        if verbose: print(f"Flow Matrix (w):\n {w}")
        print(f"Number of nodes(n): {n}")
        print(f"Fixed hub costs (f): {f}")
        print(f"Originating flows (q): {q}")
        print(f"Destination flows (d): {d}")
        print("")

    return n, c, w, f, q, d

if __name__ == "__main__":
    small_network_data = load_data_prefixed(prefix="SMALL", verbose=True)
    large_network_data = load_data_prefixed(prefix="LARGE", verbose=True)
