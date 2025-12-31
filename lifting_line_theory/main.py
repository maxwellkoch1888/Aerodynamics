import json
import numpy as np
import matplotlib.pyplot as plt

# Open the json file
with open("input.json","r") as file:
    data = json.load(file)

#Define the aspect ratio and taper ratio
Ra = data["wing"]["planform"]["aspect_ratio"]
Rt = data["wing"]["planform"]["taper_ratio"]

# Get number of nodes
N = 1 + 2*data["wing"]["nodes_per_semispan"]
c_la = data["wing"]["airfoil_lift_slope"]

# Define coordinate change
theta = np.linspace(0,np.pi,N)
z_b = -0.5*np.cos(theta)

# Define the chord length at the root
c_b = 2*(1-(1-Rt)*np.abs(np.cos(theta)))/Ra/(1+Rt)

# Use eq 6.31 and 6.30 to calculate omega for the given washout distribution
omega = 0
washout_distribution = data["wing"]["washout"]["distribution"]
if washout_distribution == "optimum":
    omega = 1-np.sin(theta)/(c_b/c_b[N//2])
elif washout_distribution == "linear":
    omega = abs(np.cos(theta))
elif washout_distribution == "none":
    omega = np.zeros_like(theta)

#Define aileron values
aileron_root = data["wing"]["aileron"]["begin[z/b]"]
aileron_tip = data["wing"]["aileron"]["end[z/b]"]
cfc_root = data["wing"]["aileron"]["begin[cf/c]"]
cfc_tip= data["wing"]["aileron"]["end[cf/c]"]
hinge_efficiency = data["wing"]["aileron"]["hinge_efficiency"]
deflection_efficiency = 1

# Calculate chi value
chi = np.zeros(N)
aileron_tip_rad = float(np.arccos(aileron_tip / (-0.5)))
aileron_root_rad = float(np.arccos(aileron_root / (-0.5)))

for i in range(N):
    if z_b[i] >= -aileron_tip:
        l_aileron_start = z_b[i]
        break
for i in range(N):
    if z_b[i] >= -aileron_root:
        l_aileron_end = z_b[i-1]
        break
for i in range(N):
    if z_b[i] >= aileron_root:
        r_aileron_start = z_b[i]
        aileron_root_chord = (-.75*c_b[i] + cfc_root*c_b[i])
        break
for i in range(N):
    if aileron_tip <= z_b[i]:
        r_aileron_end = z_b[i-1]
        aileron_tip_chord = (-.75*c_b[i] + cfc_tip*c_b[i])
        break


right_aileron_xvalues = [r_aileron_start, r_aileron_end]
left_aileron_xvalues = [l_aileron_start, l_aileron_end]
right_aileron_yvalues = [aileron_root_chord, aileron_tip_chord]
left_aileron_yvalues = [aileron_tip_chord, aileron_root_chord]
chi_right_aileron_yvalues = [cfc_root, cfc_tip]
chi_left_aileron_yvalues = [cfc_tip, cfc_root]

aileron_plot = np.zeros(N)
for i in range(N):
    if aileron_root <= z_b[i] <= aileron_tip:
        aileron_plot[i] = np.interp(z_b[i], right_aileron_xvalues, right_aileron_yvalues)
    elif -aileron_root >= z_b[i] >= -aileron_tip:
        aileron_plot[i] = np.interp(z_b[i], left_aileron_xvalues, left_aileron_yvalues)
    else:
        aileron_plot[i] = None

aileron_chi = np.zeros(N)
for i in range(N):
    if aileron_root <= z_b[i] <= aileron_tip:
        aileron_chi[i] = np.interp(z_b[i], right_aileron_xvalues, chi_right_aileron_yvalues)
    elif -aileron_root >= z_b[i] >= -aileron_tip:
        aileron_chi[i] = np.interp(z_b[i], left_aileron_xvalues, chi_left_aileron_yvalues)
    else:
        aileron_chi[i] = None

for i in range(N):
    theta_i = np.arccos(z_b[i]/(-0.5))
    if np.pi-aileron_tip_rad <= theta_i <= np.pi-aileron_root_rad:
        # Interpolate the local flap chord length per wing chord length assuming a linear change in aileron chord length
        cfc_local = aileron_chi[i]
        # Calculate the local theta f value
        theta_f = np.arccos(2*abs(cfc_local) -1)
        # Calculate the local flap effectiveness
        flap_effectiveness = 1 - (theta_f - np.sin(theta_f)) / np.pi
        # Calculate local ef value assuming a linear change in aileron chord length
        ef = hinge_efficiency * deflection_efficiency * flap_effectiveness
        chi[i] = ef
    # Repeat the above process for the other aileron
    elif aileron_tip_rad >= theta_i >= aileron_root_rad:
        cfc_local = aileron_chi[i]
        theta_f = np.arccos(2*abs(cfc_local) -1)
        flap_effectiveness = 1 - (theta_f - np.sin(theta_f)) / np.pi
        ef = hinge_efficiency * deflection_efficiency * flap_effectiveness
        chi[i] = -ef

# Calculate C matrix
C = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        n = j+1
        if i == 0:
            C[i,j] = n**2
        elif i == N-1:
            C[i,j] = ((-1)**(n+1))*(n**2)
        else:
            C[i,j] = ((4/c_la/c_b[i] + n/np.sin(theta[i]))*np.sin(n*theta[i]))
C_inv = np.linalg.inv(C)
a = np.matmul(C_inv, np.ones(N))
b = np.matmul(C_inv, omega)
c = np.matmul(C_inv, chi)
d = np.matmul(C_inv, np.ones(N)*np.cos(theta))

# Calculate kappa d, kappa l, kappa dl, kappa d omega, and epsilon omega
k_d = 0
for i in range(1,N):
    n = i+1
    a_ratio = a[i]/a[0]
    k_d += n*a_ratio**2

k_l = (1- (1 + np.pi * Ra/c_la) * a[0]) / ((1 + np.pi * Ra/c_la) * a[0])

k_dl_sum = 0
for i in range(1,N):
    n = i+1
    # Check if washout is equal to zero
    if b[0] == 0:
        k_dl_sum += n * (a[i] / a[0]) * (-a[i] / a[0])
    else:
        k_dl_sum += n * (a[i]/a[0]) * (b[i]/b[0] - a[i]/a[0])
k_dl = 2 * (b[0]/a[0]) * k_dl_sum

k_do_sum = 0
for i in range(1,N):
    n = i+1
    # Check if washout is equal to zero
    if b[0] == 0:
        k_do_sum += n * ((-a[i]/a[0])**2)
    else:
        k_do_sum += n * ((b[i] / b[0] - a[i] / a[0]) ** 2)
k_do = ((b[0]/a[0])**2) * k_do_sum

eo = b[0]/a[0]

# Calculate C_la (Actual lift coefficient slope)
C_la = c_la / ((1 + c_la/(np.pi*Ra))*(1+k_l))

# Calculate Cℓ,𝛿𝑎 using 6.24
c_l_deflection = -np.pi*Ra*c[1]/4

# Calculate Cℓ,𝑝- using 6.25
c_l_roll = -np.pi*Ra*d[1]/4

# define washout magnitude
washout_mag = 0
magnitude_type = data["wing"]["washout"]["amount[deg]"]
if isinstance(magnitude_type, (int, float)):
    washout_mag = np.radians(data["wing"]["washout"]["amount[deg]"])
elif magnitude_type == "optimum":
    c_ld = data["wing"]["washout"]["CL_design"]
    washout_mag = k_dl * c_ld / (2 * k_do * c_la)

# Define conditions for alpha root, pbar, and aileron deflection
alpha_root = data["condition"]["alpha_root[deg]"]
if isinstance(alpha_root, (int, float)):
    alpha_root = np.radians(data["condition"]["alpha_root[deg]"])
elif alpha_root == "CL":
    CL_oper = data["condition"]["CL"]
    alpha_root = CL_oper/C_la + eo*washout_mag

oper_roll_rate = data["condition"]["pbar"]
aileron_deflection = np.radians(data["condition"]["aileron_deflection[deg]"])
steady_roll_rate = -(c_l_deflection/c_l_roll)*aileron_deflection
roll_rate = 0
if isinstance(oper_roll_rate, (int, float)):
    roll_rate = data["condition"]["pbar"]
elif oper_roll_rate == "steady":
    roll_rate = steady_roll_rate

# Define A vector for computation in part 9
A = np.zeros(N)
for i in range(N):
    A[i] = a[i]*alpha_root - b[i]*washout_mag + c[i]*aileron_deflection + d[i]*roll_rate

# Define C_L, C_Di, C_l, C_n, and p steady for part 9
# Using 6.5
C_L = float(np.pi * Ra * A[0])

# Using 6.6
C_Di_sum = 0
for i in range(N):
    n = i + 1
    C_Di_sum += n*A[i]**2
C_Di = np.pi*Ra*C_Di_sum - (np.pi*Ra*roll_rate)*A[1]/2

# Calculate c_l using 6.23
c_l = c_l_roll*roll_rate + c_l_deflection*aileron_deflection

# Using 1.8.73 from textbook for a wing of arbitrary geometry
C_n_sum = 0
for i in range(3,N):
    n = i + 1
    C_n_sum += (2*n-1)*A[i-1]*A[i]
C_n = C_L*(6*A[1]-roll_rate)/8 + np.pi*Ra*(10*A[1]-roll_rate)*A[2]/8 + np.pi*Ra*C_n_sum/4

# Open a file for writing
with open("output_matrices.txt", "w") as file:
    # Write C matrix
    file.write("C Matrix:\n")
    np.savetxt(file, C, fmt="%.6f")
    file.write("\n")  # Add a newline for spacing

    # Write C_inv matrix
    file.write("C Inverse Matrix:\n")
    np.savetxt(file, C_inv, fmt="%.6f")
    file.write("\n")

    # Write an
    file.write("an Fourier Coefficient:\n")
    np.savetxt(file, a.reshape(1, -1), fmt="%.6f")  # Reshape to write in a single line
    file.write("\n")

    # Write bn
    file.write("bn Fourier Coefficient:\n")
    np.savetxt(file, b.reshape(1, -1), fmt="%.6f")
    file.write("\n")

    # Write cn
    file.write("cn Fourier Coefficient:\n")
    np.savetxt(file, c.reshape(1, -1), fmt="%.6f")
    file.write("\n")

    # Write dn
    file.write("dn Fourier Coefficient:\n")
    np.savetxt(file, d.reshape(1, -1), fmt="%.6f")
    file.write("\n")

# Print required values
values = [
    ("k_L", k_l),
    ("C_L,a", C_la),
    ("ε_Ω", eo),
    ("k_D", k_d),
    ("k_DL", k_dl),
    ("k_dΩ", k_do),
    ("Cℓ,𝛿𝑎", c_l_deflection),
    ("Cℓ,𝑝-", c_l_roll),
    ("C_L", C_L),
    ("C_Di", C_Di),
    ("C_ℓ", c_l),
    ("C_𝑛", C_n),
    ("p steady", steady_roll_rate)]

for label, value in values:
    print(label)
    print(value)
    print()

# Plot wing geometry and aileron geometry
# Set the graph boundaries
ymin = -0.4
ymax = 0.3
ytotal = ymax-ymin
plt.figure(figsize=(8, 4))
plt.xlim(-0.6,0.6)
plt.ylim(ymin,ymax)
# Plot the leading edge
plt.plot(z_b, 0.25*c_b, color = 'black', linestyle = '-')
# Plot the trailing edge
plt.plot(z_b, -0.75*c_b, color = 'black', linestyle = '-')
# Plot the aileron hinge
plt.plot(z_b, aileron_plot, color = 'black')
# Label the planform and aileron locations
plt.plot(-1,-1, color = 'r', label = "Aileron Geometry")
plt.plot(-1,-1, color = 'b', label = "Wing Planform")
# Label the plot
plt.xlabel("Spanwise Position (z/b)")
plt.ylabel("Chord Position (c/b)")
plt.title("Planform")

leading_edge = np.zeros(N)
trailing_edge = np.zeros(N)
hinge_line = np.zeros(N)
for i in range(N):
    # Calculate the normalized leading edge, trailing edge, and hinge line values to define locations to stop vertical lines
    leading_edge[i] = 1-(ymax - c_b[i]/4)/ytotal
    trailing_edge[i] = (-3*c_b[i]/4-ymin)/ytotal
    hinge_line[i] = (aileron_plot[i]-ymin)/ytotal
    # Plot the vertical lines for the planform and ailerons
    if aileron_root <= z_b[i] <= aileron_tip or -aileron_root >= z_b[i] >= -aileron_tip:
        plt.axvline(x=z_b[i], ymin=hinge_line[i], ymax=leading_edge[i], color='b')
        plt.axvline(x=z_b[i], ymin=trailing_edge[i], ymax=hinge_line[i], color='r')
    else:
        plt.axvline(x=z_b[i], ymin=trailing_edge[i], ymax=leading_edge[i], color='b')

plt.legend()
plt.show()

# Plot the washout distribution
plt.plot(z_b, omega, color = 'black')
plt.xlabel("Spanwise Position (z/b)")
plt.ylabel("Omega")
plt.title("Washout Distribution")
plt.show()

# Plot the aileron distribution
plt.plot(z_b, chi, color = 'black')
plt.xlabel("Spanwise Position (z/b)")
plt.ylabel("Chi")
plt.title("Aileron Distribution")
plt.show()

# plot the CL hat plot
# CL planform effect
CL_planform_sum = 0
for i in range(N):
    n = i+1
    CL_planform_sum += a[i]*np.sin(n*theta)
CL_planform = 4*alpha_root*CL_planform_sum

# CL washout effect
CL_washout_sum = 0
for i in range(N):
    n = i+1
    CL_washout_sum += b[i]*np.sin(n*theta)
CL_washout = -4*washout_mag*CL_washout_sum

# CL aileron effect
CL_aileron_sum = 0
for i in range(N):
    n = i+1
    CL_aileron_sum += c[i]*np.sin(n*theta)
CL_aileron = 4*aileron_deflection*CL_aileron_sum

# CL roll effect
CL_roll_sum = 0
for i in range(N):
    n = i+1
    CL_roll_sum += d[i]*np.sin(n*theta)
CL_roll = 4*roll_rate*CL_roll_sum

CL_hat = CL_planform + CL_washout + CL_aileron + CL_roll

# plt.plot(z_b, CL_planform, color = 'b', label = "Planform")
# plt.plot(z_b, CL_washout, color = 'orange', label = "Washout")
# plt.plot(z_b, CL_aileron, color = 'g', label = "Aileron")
# plt.plot(z_b, CL_roll, color = 'r', label = "Roll")
# plt.plot(z_b, CL_hat, color = 'purple', label = "Total" )
# plt.xlabel("Spanwise Position (z/b)")
# plt.ylabel("CL Hat")
# plt.title("CL Hat Distributions")
# plt.legend()
# 
# plt.show()

# Plot the CL tilde distribution
# CL planform effect
CLt_planform = CL_planform/c_b

# CL washout effect
CLt_washout = CL_washout/c_b

# CL aileron effect
CLt_aileron = CL_aileron/c_b

# CL roll effect
CLt_roll = CL_roll/c_b

CL_tilde = CLt_planform + CLt_washout + CLt_aileron + CLt_roll

plt.plot(z_b, CLt_planform, color = 'b', label = "Planform")
plt.plot(z_b, CLt_washout, color = 'orange', label = "Washout")
plt.plot(z_b, CLt_aileron, color = 'g', label = "Aileron")
plt.plot(z_b, CLt_roll, color = 'r', label = "Roll")
plt.plot(z_b, CL_tilde, color = 'purple', label = "Total" )
plt.xlabel("Spanwise Position (z/b)")
plt.ylabel("CL Tilde")
plt.title("CL Tilde Distributions")
plt.legend()

plt.show()




# Section added for senior design, aileron ideal location
section_moment = []
for i in range(N):
    section_moment.append(CL_tilde[i] * z_b[i])

# Find the maximum and minimum section moments
max_section_moment = max(section_moment)
min_section_moment = min(section_moment)

# Find the z_b values corresponding to the max and min section moments
max_index = section_moment.index(max_section_moment)
min_index = section_moment.index(min_section_moment)

z_b_max = z_b[max_index]
z_b_min = z_b[min_index]

aileron_span = 0
aileron_root_end = max_index-1
aileron_tip_start = max_index+1
while aileron_span < 0.2:
    if section_moment[aileron_root_end] > section_moment[aileron_tip_start]:
        aileron_root_end = aileron_root_end - 1
    if section_moment[aileron_root_end] < section_moment[aileron_tip_start]:
        aileron_tip_start = aileron_tip_start + 1
    aileron_span = z_b[aileron_tip_start] - z_b[aileron_root_end]
    


# # Print the results
# print("Max section moment:", max_section_moment)
# print("Ideal right aileron center:", z_b_max)
# print()
# print("Aileron start:", z_b[aileron_tip_start])
# print()
# print("Aileron end:", z_b[aileron_root_end])




# # Plot the results
# plt.plot(z_b, section_moment)
# plt.xlabel("Spanwise Position")
# plt.ylabel("Local Section Moment")
# plt.title("Section Moment vs Span")
# plt.show()

