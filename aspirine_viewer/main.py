import pandas as pd
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np


with open('./__files/aspirin.xyz') as readfile:
    for idx, line in enumerate(readfile):
        if idx == 0:
            number_of_atoms = line.strip()
        if idx == 1:
            comment = line.strip()

# read values into dataframe
df = pd.read_csv("./__files/aspirin.xyz", sep="\t")
df = df.iloc[1:, :]
df = df[df.columns[0]].str.split("      ", expand=True)
df.columns = ['Atomic species', 'x', 'y', 'z']

# s: square of intended radius for each atom according to atomic species
df.loc[(df['Atomic species'] == 'H'), 's'] = '250'
df.loc[(df['Atomic species'] == 'O'), 's'] = '360'
df.loc[(df['Atomic species'] == 'C'), 's'] = '490'
float_dfs = [float(s) for s in df['s']]

df.loc[(df['Atomic species'] == 'H'), 'c'] = 'blue'
df.loc[(df['Atomic species'] == 'O'), 'c'] = 'purple'
df.loc[(df['Atomic species'] == 'C'), 'c'] = 'black'

# convert df values to floats in a list
x_float = [float(x) for x in df['x']]
y_float = [float(y) for y in df['y']]
z_float = [float(z) for z in df['z']]

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter3D(x_float, y_float, z_float, s=float_dfs, c=df['c'])
ax.set(title='3D Plot Aspirin Atoms')

# extract bonds
matrix = np.array([x_float, y_float, z_float])
matrix = matrix.T
dist_matrix = sp.spatial.distance_matrix(matrix, matrix)
args = np.argwhere(dist_matrix < 1.6)
bonds = []

for j in range(args.shape[0]):
    start = args[j][0]
    end = args[j][1]
    start_pt = np.array([x_float[start], y_float[start], z_float[start]])
    end_pt = np.array([x_float[end], y_float[end], z_float[end]])
    if all(start_pt != end_pt):
        bond = np.array([start_pt, end_pt])
        bonds.append(bond)

bonds_array = np.stack(bonds)
conn_bonds = Line3DCollection(
    bonds_array, edgecolor='grey', linestyle='solid', linewidth=3)
ax.add_collection3d(conn_bonds)
plt.savefig('plot.pdf')
# plt.show()
