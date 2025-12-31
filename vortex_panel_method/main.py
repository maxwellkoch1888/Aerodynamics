import numpy as np
import json
from tabulate import tabulate
from pathlib import Path
import matplotlib.pyplot as plt

# Directory containing this script
BASE_DIR = Path(__file__).parent

# Open all needed json files for the airfoils being examined.
with open(BASE_DIR / '10data.json', 'r') as file10:
    data10 = json.load(file10)

geometry10 = data10["geometry"]
v10 = data10["freestream_velocity"]

with open(BASE_DIR / '200data.json', 'r') as file2412200:
    data2412200 = json.load(file2412200)

geometry2412200 = data2412200["geometry"]
v2412200 = data2412200["freestream_velocity"]

with open(BASE_DIR / '2421data.json', 'r') as file2421:
    data2421 = json.load(file2421)

geometry2421 = data2421["geometry"]
v2421 = data2421["freestream_velocity"]

with open(BASE_DIR / '0015data.json', 'r') as file0015:
    data0015 = json.load(file0015)

geometry0015 = data0015["geometry"]
v0015 = data0015["freestream_velocity"]

# Set the range of angles of attack that will be determined.
range_alpha = np.arange(-12, 17, 2)

# Builds a function that outputs cL, cMLE, and cMC4 by calculating the gamma vector
def coef(geometry, V, alpha):
    alpha = np.radians(alpha)

    geom_path = Path(geometry)
    if not geom_path.is_absolute():
        geom_path = BASE_DIR / geom_path

    n_pts = np.loadtxt(geom_path, dtype=float)
    # specify points, control points, dx, dy, length, and applied conditions
    n = n_pts.shape[0]
    c_pts = 0.5 * (n_pts[1:] + n_pts[:-1])
    dx = (n_pts[1:, 0] - n_pts[:-1, 0])
    dy = (n_pts[1:, 1] - n_pts[:-1, 1])
    l = np.sqrt(dx ** 2 + dy ** 2)

    # Build the A matrix
    A = np.zeros((n, n))
    for i in range(n - 1):
        for j in range((n - 1)):
            xi = (dx[j] * (c_pts[i, 0] - n_pts[j, 0]) + dy[j] * (c_pts[i, 1] - n_pts[j, 1])) / l[j]
            eta = (-dy[j] * (c_pts[i, 0] - n_pts[j, 0]) + dx[j] * (c_pts[i, 1] - n_pts[j, 1])) / l[j]

            phi = np.arctan2(eta * l[j], eta ** 2 + xi ** 2 - xi * l[j])
            psi = 0.5 * np.log((xi ** 2 + eta ** 2) / ((xi - l[j]) ** 2 + eta ** 2))

            P1 = np.array([[dx[j], -dy[j]], [dy[j], dx[j]]])
            P2 = np.array([[(l[j] - xi) * phi + eta * psi, xi * phi - eta * psi],
                           [eta * phi - (l[j] - xi) * psi - l[j], -eta * phi - xi * psi + l[j]]])
            P = (1 / (2 * np.pi * l[j] ** 2)) * np.matmul(P1, P2)

            A[i, j] += dx[i] * P[1, 0] / l[i] - dy[i] * P[0, 0] / l[i]
            A[i, j + 1] += dx[i] * P[1, 1] / l[i] - dy[i] * P[0, 1] / l[i]

    A[n - 1, 0] = 1.0
    A[n - 1, n - 1] = 1.0

    # Build the B matrix
    B = np.zeros((n, 1))  # fill in with rest of values, compute b values
    for j in range(n - 1):
        P = V * (dy[j] * np.cos(alpha) - dx[j] * np.sin(alpha)) / l[j]
        B[j] += P

    # Solve for gamma
    gamma = np.linalg.solve(A, B)

    # Calculate cl, cmle, cmc/4 using the equations from page 31 of the textbook and the notes from class
    cL = 0
    for j in range(n - 1):
        cLpan = l[j] / V * (gamma[j] + gamma[j + 1])
        cL = cL + cLpan

    cMLE = 0
    for j in range(n - 1):
        cMLEpan = l[j] * ((2 * n_pts[j, 0] * gamma[j] +
                           n_pts[j, 0] * gamma[j + 1] +
                           n_pts[j + 1, 0] * gamma[j] +
                           2 * n_pts[j + 1, 0] * gamma[j + 1]) / V * np.cos(alpha)
                          + (2 * n_pts[j, 1] * gamma[j] +
                             n_pts[j, 1] * gamma[j + 1] +
                             n_pts[j + 1, 1] * gamma[j] +
                             2 * n_pts[j + 1, 1] * gamma[j + 1]) / V * np.sin(alpha))
        cMLE += -cMLEpan / 3

    cMC4 = cMLE + cL * .25

    return cL, cMLE, cMC4


# Builds a matrix that compiles the results of cL, cMLE, and cMC4
# This matrix can later be used to build a table of values or plot figures
def results(geometry, V):
    # build four lists that will store values for angle of attack, cL, cMLE, and cMC4
    alpha_values = []
    cL_values = []
    cMLE_values = []
    cMC4_values = []
    # append the results for each angle of attack to its respective list
    for alpha_deg in range_alpha:
        cL, cMLE, cMC4 = coef(geometry, V, alpha_deg)
        alpha_values.append(float(alpha_deg))
        cL_values.append(float(cL))
        cMLE_values.append(float(cMLE))
        cMC4_values.append(float(cMC4))
    # output a dictionary with results from each angle of attack
    return {'alpha': np.array(alpha_values),
            'cL': np.array(cL_values),
            'cMLE': np.array(cMLE_values),
            'cMC4': np.array(cMC4_values)}


# Builds a table of values using a given result matrix.
def table(geometry, V):
    data = results(geometry, V)
    table_data = list(zip(data['alpha'], data['cL'], data['cMLE'], data['cMC4']))
    headers = ["Angle (degrees)", "cL", "cMLE", "cMC4"]
    table = tabulate(table_data, headers=headers)
    print(table)


# print necessary tables and plots
print("2412 Airfoil with 10 Nodes")
table(geometry10, v10)
print()
print("2412 Airfoil with 200 Nodes")
table(geometry2412200, v2412200)
print()
print("2421 Airfoil with 200 Nodes")
table(geometry2421, v2421)
print()
print("0015 Airfoil with 200 Nodes")
table(geometry0015, v0015)

geom_2412200_path = Path(geometry2412200)
if not geom_2412200_path.is_absolute():
    geom_2412200_path = BASE_DIR / geom_2412200_path

pts_2412200 = np.loadtxt(geom_2412200_path, dtype=float)

x = pts_2412200[:, 0]
y = pts_2412200[:, 1]

plt.figure()
plt.plot(x, y, marker='o', linestyle='-')
plt.axis('equal')
plt.xlabel("x/c")
plt.ylabel("y/c")
plt.title("NACA 2412 Airfoil Geometry (200 Nodes)")
plt.grid(True)
plt.show()