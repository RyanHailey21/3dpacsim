import aerosandbox as asb
import aerosandbox.numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
DROP_HEIGHT = 18.8     # meters (Starting altitude)
MIN_THICKNESS = 0.10   # 10% Minimum thickness (Structural requirement)
PAYLOAD_MASS = 0.200   # kg (Battery, Flight Controller, Servo)

# 3D Printing Material Props (ASA)
RHO_ASA = 1070.0       # kg/m^3
WALL_THICKNESS = 0.0004 * 2  # 2 perimeters (0.8mm total shell)
INFILL_PCT = 0.05      # 5% Infill

# 1. Initialize the Optimizer
opti = asb.Opti()

# --- DESIGN VARIABLES ---
# Geometry
wingspan = opti.variable(init_guess=1.0, lower_bound=0.6, upper_bound=1)
chord = opti.variable(init_guess=0.2, lower_bound=0.12, upper_bound=0.35)
tail_arm = opti.variable(init_guess=0.6, lower_bound=0.4, upper_bound=1.0)
tail_area = opti.variable(init_guess=0.04, lower_bound=0.02, upper_bound=0.1)

# Flight State (Variables to be found)
alpha = opti.variable(init_guess=4, lower_bound=0, upper_bound=10)
velocity = opti.variable(init_guess=9, lower_bound=5, upper_bound=25)

# Airfoil Shape (Kulfan Parameters)
# Initialized to a generic thick shape to aid convergence
upper_init = np.array([0.2, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15])
lower_init = np.array([-0.1, -0.1, -0.05, -0.05, -0.05, -0.02, -0.02, -0.01])

upper_weights = opti.variable(init_guess=upper_init, lower_bound=0.01, upper_bound=0.5)
lower_weights = opti.variable(init_guess=lower_init, lower_bound=-0.5, upper_bound=-0.01)

# --- AERODYNAMICS OBJECTS ---
airfoil = asb.KulfanAirfoil(
    name="Optimized_Airfoil",
    upper_weights=upper_weights,
    lower_weights=lower_weights,
    leading_edge_weight=0.0,
    TE_thickness=0.002
)
tail_airfoil = asb.Airfoil("naca0008")

# --- GEOMETRIC CONSTRAINTS ---
# 1. Spar Constraint: Airfoil must be thick enough at 30% chord
thickness_at_spar = airfoil.local_thickness(x_over_c=0.30)
opti.subject_to(thickness_at_spar >= MIN_THICKNESS)

# --- MASS MODEL (Physics-Based) ---
# Calculate real geometric properties
wing_area_planform = wingspan * chord
# Kulfan area is non-dimensional. Real Area = (A_nd) * c^2
airfoil_cross_section_area = airfoil.area() * (chord**2)
wing_volume_internal = airfoil_cross_section_area * wingspan

# Mass Summation
mass_wing = (wing_area_planform * WALL_THICKNESS * RHO_ASA) + \
            (wing_volume_internal * INFILL_PCT * RHO_ASA)

mass_tail = tail_area * WALL_THICKNESS * RHO_ASA
mass_boom = tail_arm * 0.06  # Carbon tube approx
total_mass = PAYLOAD_MASS + mass_wing + mass_tail + mass_boom

# --- AIRCRAFT DEFINITION ---
wing = asb.Wing(
    name="Main Wing",
    xsecs=[
        asb.WingXSec(xyz_le=[0, -wingspan/2, 0], chord=chord, airfoil=airfoil),
        asb.WingXSec(xyz_le=[0, wingspan/2, 0], chord=chord, airfoil=airfoil)
    ],
    symmetric=False # Explicit geometry for easier export
)

h_tail = asb.Wing(
    name="Horizontal Tail",
    xsecs=[
        asb.WingXSec(xyz_le=[tail_arm, -tail_area**0.5/2, 0], chord=tail_area**0.5, airfoil=tail_airfoil),
        asb.WingXSec(xyz_le=[tail_arm, tail_area**0.5/2, 0], chord=tail_area**0.5, airfoil=tail_airfoil)
    ],
    symmetric=False
)

airplane = asb.Airplane(wings=[wing, h_tail], c_ref=chord)

# --- AERODYNAMIC ANALYSIS ---
op_point = asb.OperatingPoint(velocity=velocity, alpha=alpha)
analysis = asb.AeroBuildup(airplane=airplane, op_point=op_point).run()

# --- PHYSICS CONSTRAINTS ---
# 1. Equilibrium (Lift = Weight)
opti.subject_to(analysis['L'] == total_mass * 9.81)

# 2. Trim (Pitching Moment = 0)
opti.subject_to(analysis['Cm'] == 0)

# 3. Static Stability (dCm/dAlpha < 0)
# Check stability by perturbing alpha slightly
op_point_perturbed = asb.OperatingPoint(velocity=velocity, alpha=alpha + 1)
analysis_perturbed = asb.AeroBuildup(airplane=airplane, op_point=op_point_perturbed).run()
dCm_dalpha = (analysis_perturbed['Cm'] - analysis['Cm']) / 1.0
opti.subject_to(dCm_dalpha <= -0.05) # Healthy stability margin

# --- OBJECTIVE: NOSE-DIVE RECOVERY TIME ---
# Energy Balance: To reach velocity V, we must lose potential energy.
# h_loss = v^2 / 2g. We add a factor (1.5) for drag losses during the dive.
height_lost_to_dive = 1.5 * (velocity**2) / (2 * 9.81)

usable_glide_height = DROP_HEIGHT - height_lost_to_dive

# Sanity Check: We must have height left to glide
opti.subject_to(usable_glide_height > 2.0)

# Sink Rate in steady glide
sink_rate = (analysis['D'] * velocity) / (total_mass * 9.81)

# Time = Height / SinkSpeed
time_gliding = usable_glide_height / sink_rate

# Maximize Time
opti.minimize(-time_gliding)

# --- SOLVE ---
print("Running Optimization...")
try:
    sol = opti.solve()
except RuntimeError:
    print("Optimization failed to converge perfectly. Using best attempt.")
    sol = opti.debug

# --- EXTRACT RESULTS ---
v_val = sol.value(velocity)
vz_val = sol.value(sink_rate)
h_loss_val = sol.value(height_lost_to_dive)
glide_time_val = sol.value(time_gliding)
mass_val = sol.value(total_mass)
span_val = sol.value(wingspan)
chord_val = sol.value(chord)
t_arm_val = sol.value(tail_arm)
t_area_val = sol.value(tail_area)
alpha_val = sol.value(alpha)
opt_airfoil = sol.value(airfoil)

# Extracting the 16 Kulfan weights
up_w = sol.value(upper_weights)
lo_w = sol.value(lower_weights)

# --- DETAILED OPTIMIZATION REPORT ---
print(f"\n" + "="*40)
print(f"       WINNING DESIGN PARAMETERS")
print(f"="*40)

print(f"--- PERFORMANCE ---")
print(f"Total Flight Time:  {glide_time_val + 1.0:.2f} s")
print(f"Glide Duration:     {glide_time_val:.2f} s")
print(f"Altitude Lost:      {h_loss_val:.2f} m (Dive phase)")
print(f"Steady Velocity:    {v_val:.2f} m/s")
print(f"Sink Rate:          {vz_val:.3f} m/s")
print(f"Angle of Attack:    {alpha_val:.2f}°")

print(f"\n--- DIMENSIONS & MASS ---")
print(f"Total Mass:         {mass_val*1000:.1f} g")
print(f"Wingspan:           {span_val:.3f} m")
print(f"Chord:              {chord_val:.3f} m")
print(f"Aspect Ratio:       {span_val/chord_val:.2f}")
print(f"Tail Arm:           {t_arm_val:.3f} m")
print(f"Tail Area:          {t_area_val:.4f} m²")

print(f"\n--- AIRFOIL GEOMETRY ---")
print(f"Max Thickness:      {opt_airfoil.max_thickness():.1%}")
print(f"Max Camber:         {opt_airfoil.max_camber():.1%}")

print(f"\nUpper Kulfan Weights:")
print("  " + ", ".join([f"{w:.4f}" for w in up_w]))

print(f"Lower Kulfan Weights:")
print("  " + ", ".join([f"{w:.4f}" for w in lo_w]))
print(f"="*40)

# --- PERFORM EXPORTS ---
coords = opt_airfoil.coordinates

# 1. Save STL of the Wing
def save_manual_stl(filename, coords, span, chord):
    """
    Creates a binary-style ASCII STL of the main wing.
    Extrudes the airfoil to the optimized span.
    """
    # Scale normalized coordinates (0..1) to actual meters
    x = coords[:, 0] * chord
    y = coords[:, 1] * chord

    # Center the wing (Span goes from -span/2 to +span/2)
    z_left = -span / 2
    z_right = span / 2

    with open(filename, 'w') as f:
        f.write(f"solid {filename}\n")

        # Iterate through points to create the "skin" of the wing
        for i in range(len(x) - 1):
            # Define vertices for two triangles forming a quad
            p1 = (x[i], y[i], z_left)
            p2 = (x[i+1], y[i+1], z_left)
            p3 = (x[i], y[i], z_right)
            p4 = (x[i+1], y[i+1], z_right)

            # Triangle 1
            f.write("facet normal 0 0 0\nouter loop\n")
            f.write(f"vertex {p1[0]:.6f} {p1[1]:.6f} {p1[2]:.6f}\n")
            f.write(f"vertex {p2[0]:.6f} {p2[1]:.6f} {p2[2]:.6f}\n")
            f.write(f"vertex {p3[0]:.6f} {p3[1]:.6f} {p3[2]:.6f}\n")
            f.write("endloop\nendfacet\n")

            # Triangle 2
            f.write("facet normal 0 0 0\nouter loop\n")
            f.write(f"vertex {p2[0]:.6f} {p2[1]:.6f} {p2[2]:.6f}\n")
            f.write(f"vertex {p4[0]:.6f} {p4[1]:.6f} {p4[2]:.6f}\n")
            f.write(f"vertex {p3[0]:.6f} {p3[1]:.6f} {p3[2]:.6f}\n")
            f.write("endloop\nendfacet\n")

        f.write(f"endsolid {filename}\n")

save_manual_stl("optimized_wing.stl", coords, span_val, chord_val)
print(f"[CAD] 3D Model saved as 'optimized_wing.stl'")


# 2. Save DXF of the Airfoil
def save_airfoil_dxf(coords, filename):
    """Saves the airfoil profile as a DXF for CAD sketching."""
    with open(filename, 'w') as f:
        f.write("0\nSECTION\n2\nENTITIES\n")
        for i in range(len(coords) - 1):
            x1, y1 = coords[i]
            x2, y2 = coords[i+1]
            f.write("0\nLINE\n8\n0\n")
            f.write(f"10\n{x1:.6f}\n20\n{y1:.6f}\n30\n0.0\n")
            f.write(f"11\n{x2:.6f}\n21\n{y2:.6f}\n31\n0.0\n")

        # Close the loop
        x_first, y_first = coords[0]
        x_last, y_last = coords[-1]
        if ((x_first - x_last)**2 + (y_first - y_last)**2)**0.5 > 1e-6:
            f.write("0\nLINE\n8\n0\n")
            f.write(f"10\n{x_last:.6f}\n20\n{y_last:.6f}\n30\n0.0\n")
            f.write(f"11\n{x_first:.6f}\n21\n{y_first:.6f}\n31\n0.0\n")

        f.write("0\nENDSEC\n0\nEOF\n")

save_airfoil_dxf(coords, "optimized_airfoil_profile.dxf")
print(f"[CAD] 2D Profile saved as 'optimized_airfoil_profile.dxf'")


# --- 3. PLOTTING ---
plt.figure(figsize=(12, 5))
plt.plot(coords[:, 0], coords[:, 1], 'k-', linewidth=2, label='Optimized Surface')
plt.fill(coords[:, 0], coords[:, 1], 'skyblue', alpha=0.4)
plt.plot([0, 1], [0, 0], 'r--', alpha=0.5, label='Chord Line')
plt.axis('equal')
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.title(
    f"Optimized Glider Airfoil\n"
    f"Max Thickness: {opt_airfoil.max_thickness():.1%} | "
    f"Span: {span_val:.2f}m | Chord: {chord_val:.2f}m"
)
plt.xlabel("x/c")
plt.ylabel("y/c")
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
