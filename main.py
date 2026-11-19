
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="SHM Physics Laboratory",
    page_icon="⚛️",
    layout="wide"
)

st.title("⚛️ SHM Simulator")
st.caption(
    "Interactive Physics Laboratory | "
    "Analytical Solution | Euler | RK4"
)

# =====================================================
# 2. SIDEBAR CONTROLS
# =====================================================

st.sidebar.title("Simulation Controls")

mass = st.sidebar.slider(
    "Mass (kg)", 0.1, 10.0, 1.0, 0.1
)

k = st.sidebar.slider(
    "Spring Constant (N/m)", 1.0, 100.0, 20.0, 1.0
)

x0 = st.sidebar.slider(
    "Initial Displacement (m)",
    -0.5, 0.5, 0.1, 0.01
)

v0 = st.sidebar.slider(
    "Initial Velocity (m/s)",
    -2.0, 2.0, 0.0, 0.05
)

dt_requested = st.sidebar.select_slider(
    "Time Step (seconds)",
    options=[0.001, 0.005, 0.01, 0.02, 0.05],
    value=0.01
)

duration = st.sidebar.slider(
    "Simulation Duration (seconds)",
    2, 20, 10
)

# =====================================================
# 3. THEORETICAL SHM CALCULATIONS
# =====================================================

omega = np.sqrt(k / mass)
period = 2 * np.pi / omega
frequency = 1 / period

steps = int(round(duration / dt_requested))
dt = duration / steps

time = np.linspace(0, duration, steps + 1)

# Analytical displacement
x_theory = (
    x0 * np.cos(omega * time)
    + (v0 / omega) * np.sin(omega * time)
)

# Analytical velocity
v_theory = (
    -x0 * omega * np.sin(omega * time)
    + v0 * np.cos(omega * time)
)

# Analytical acceleration
a_theory = -(k / mass) * x_theory

# =====================================================
# 4. EULER NUMERICAL METHOD
# =====================================================

x_euler = np.zeros(steps + 1)
v_euler = np.zeros(steps + 1)

x_euler[0] = x0
v_euler[0] = v0

for i in range(steps):

    acceleration = -(k / mass) * x_euler[i]

    x_euler[i + 1] = (
        x_euler[i] + v_euler[i] * dt
    )

    v_euler[i + 1] = (
        v_euler[i] + acceleration * dt
    )

a_euler = -(k / mass) * x_euler

# =====================================================
# 5. RUNGE-KUTTA 4TH ORDER METHOD
# =====================================================

def derivatives(x, v):
    dx = v
    dv = -(k / mass) * x
    return dx, dv


x_rk4 = np.zeros(steps + 1)
v_rk4 = np.zeros(steps + 1)

x_rk4[0] = x0
v_rk4[0] = v0

for i in range(steps):

    x = x_rk4[i]
    v = v_rk4[i]

    k1x, k1v = derivatives(x, v)

    k2x, k2v = derivatives(
        x + dt * k1x / 2,
        v + dt * k1v / 2
    )

    k3x, k3v = derivatives(
        x + dt * k2x / 2,
        v + dt * k2v / 2
    )

    k4x, k4v = derivatives(
        x + dt * k3x,
        v + dt * k3v
    )

    x_rk4[i + 1] = x + (
        dt / 6
    ) * (
        k1x + 2*k2x + 2*k3x + k4x
    )

    v_rk4[i + 1] = v + (
        dt / 6
    ) * (
        k1v + 2*k2v + 2*k3v + k4v
    )

a_rk4 = -(k / mass) * x_rk4

# =====================================================
# 6. SYSTEM PARAMETERS DASHBOARD
# =====================================================

st.subheader("System Parameters")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Angular Frequency",
    f"{omega:.4f} rad/s"
)

c2.metric(
    "Time Period",
    f"{period:.4f} s"
)

c3.metric(
    "Frequency",
    f"{frequency:.4f} Hz"
)

initial_energy = (
    0.5 * mass * v0**2
    + 0.5 * k * x0**2
)

c4.metric(
    "Initial Energy",
    f"{initial_energy:.5f} J"
)

st.caption(
    f"Actual integration time step: {dt:.5f} seconds"
)

st.divider()

# =====================================================
# 7. MOTION ANALYSIS GRAPHS
# =====================================================

st.subheader("Motion Analysis")

fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    subplot_titles=(
        "Displacement vs Time",
        "Velocity vs Time",
        "Acceleration vs Time"
    )
)

graph_data = [
    (x_theory, x_euler, x_rk4, "Displacement (m)"),
    (v_theory, v_euler, v_rk4, "Velocity (m/s)"),
    (a_theory, a_euler, a_rk4, "Acceleration (m/s²)")
]

for row, (theory, euler, rk4, label) in enumerate(
    graph_data, start=1
):

    fig.add_trace(
        go.Scatter(
            x=time,
            y=theory,
            name="Theory",
            line=dict(color="#38BDF8", width=2),
            showlegend=(row == 1)
        ),
        row=row,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=time,
            y=euler,
            name="Euler",
            line=dict(color="#F97316", dash="dash"),
            showlegend=(row == 1)
        ),
        row=row,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=time,
            y=rk4,
            name="RK4",
            line=dict(color="#A78BFA", dash="dot"),
            showlegend=(row == 1)
        ),
        row=row,
        col=1
    )

    fig.update_yaxes(
        title_text=label,
        row=row,
        col=1
    )

fig.update_xaxes(
    title_text="Time (seconds)",
    row=3,
    col=1
)

fig.update_layout(
    template="plotly_dark",
    height=750,
    hovermode="x unified"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# 8. ENERGY ANALYSIS
# =====================================================

st.divider()
st.subheader("Energy Conservation")

# Theoretical energy
theory_ke = 0.5 * mass * v_theory**2
theory_pe = 0.5 * k * x_theory**2
theory_energy = theory_ke + theory_pe

# Euler energy
euler_ke = 0.5 * mass * v_euler**2
euler_pe = 0.5 * k * x_euler**2
euler_energy = euler_ke + euler_pe

# RK4 energy
rk4_ke = 0.5 * mass * v_rk4**2
rk4_pe = 0.5 * k * x_rk4**2
rk4_energy = rk4_ke + rk4_pe

energy_fig = go.Figure()

energy_fig.add_trace(
    go.Scatter(
        x=time,
        y=rk4_ke,
        name="Kinetic Energy",
        line=dict(color="#38BDF8")
    )
)

energy_fig.add_trace(
    go.Scatter(
        x=time,
        y=rk4_pe,
        name="Potential Energy",
        line=dict(color="#F97316")
    )
)

energy_fig.add_trace(
    go.Scatter(
        x=time,
        y=rk4_energy,
        name="Total Energy (RK4)",
        line=dict(color="#22C55E", width=2)
    )
)

energy_fig.add_trace(
    go.Scatter(
        x=time,
        y=theory_energy,
        name="Theoretical Energy",
        line=dict(color="#A78BFA", dash="dash")
    )
)

energy_fig.update_layout(
    template="plotly_dark",
    height=450,
    xaxis_title="Time (seconds)",
    yaxis_title="Energy (J)",
    hovermode="x unified"
)

st.plotly_chart(
    energy_fig,
    use_container_width=True
)

# =====================================================
# 9. ANIMATED SPRING-MASS SYSTEM
# =====================================================

st.divider()
st.subheader("🎬 Live Spring-Mass Simulation")
st.write(
    "Observe the mass oscillating around its equilibrium position. "
    "This original 2D animation is retained."
)

wall_x = -1.8
equilibrium_x = 1.0
block_width = 0.45

sample_indices = np.unique(
    np.linspace(0, steps, min(180, steps + 1), dtype=int)
)

def spring_coordinates(mass_position):
    block_left = mass_position - block_width / 2
    spring_x = np.linspace(wall_x + 0.08, block_left, 160)
    phase = np.linspace(0, 16 * np.pi, 160)
    spring_y = 0.14 * np.sin(phase)
    return spring_x, spring_y

first_index = sample_indices[0]
first_position = equilibrium_x + 2.0 * x_rk4[first_index]
first_spring_x, first_spring_y = spring_coordinates(first_position)

animation_fig = go.Figure()

animation_fig.add_trace(
    go.Scatter(
        x=first_spring_x,
        y=first_spring_y,
        mode="lines",
        line=dict(color="#38BDF8", width=4),
        name="Spring"
    )
)

animation_fig.add_trace(
    go.Scatter(
        x=[first_position],
        y=[0],
        mode="markers",
        marker=dict(
            symbol="square",
            size=30 + 5 * np.sqrt(mass),
            color="#A78BFA",
            line=dict(color="white", width=2)
        ),
        name="Mass"
    )
)

animation_fig.add_trace(
    go.Scatter(
        x=[equilibrium_x, equilibrium_x],
        y=[-0.55, 0.55],
        mode="lines",
        line=dict(color="#FACC15", dash="dash", width=2),
        name="Equilibrium"
    )
)

animation_fig.add_trace(
    go.Scatter(
        x=[wall_x, wall_x],
        y=[-0.4, 0.4],
        mode="lines",
        line=dict(color="#94A3B8", width=8),
        name="Wall"
    )
)

frames = []

for frame_number, index in enumerate(sample_indices):
    mass_position = equilibrium_x + 2.0 * x_rk4[index]
    spring_x, spring_y = spring_coordinates(mass_position)

    frames.append(
        go.Frame(
            data=[
                go.Scatter(x=spring_x, y=spring_y),
                go.Scatter(
                    x=[mass_position],
                    y=[0],
                    marker=dict(size=30 + 5 * np.sqrt(mass))
                )
            ],
            traces=[0, 1],
            name=str(frame_number)
        )
    )

animation_fig.frames = frames

animation_fig.update_layout(
    template="plotly_dark",
    height=480,
    title="Simple Harmonic Motion - Spring Mass System",
    xaxis=dict(
        range=[-2.2, 2.5],
        title="Horizontal Position",
        zeroline=False
    ),
    yaxis=dict(range=[-0.8, 0.8], visible=False),
    hovermode=False,
    updatemenus=[
        dict(
            type="buttons",
            direction="left",
            x=0.05,
            y=1.16,
            buttons=[
                dict(
                    label="▶ Play",
                    method="animate",
                    args=[
                        None,
                        {
                            "frame": {"duration": 30, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 0}
                        }
                    ]
                ),
                dict(
                    label="⏸ Pause",
                    method="animate",
                    args=[
                        [None],
                        {
                            "mode": "immediate",
                            "frame": {"duration": 0, "redraw": False},
                            "transition": {"duration": 0}
                        }
                    ]
                ),
                dict(
                    label="↺ Reset",
                    method="animate",
                    args=[
                        ["0"],
                        {
                            "mode": "immediate",
                            "frame": {"duration": 0, "redraw": True},
                            "transition": {"duration": 0}
                        }
                    ]
                )
            ]
        )
    ],
    sliders=[
        dict(
            active=0,
            currentvalue=dict(prefix="Simulation time: "),
            steps=[
                dict(
                    label=f"{time[sample_indices[j]]:.2f}s",
                    method="animate",
                    args=[
                        [str(j)],
                        {
                            "mode": "immediate",
                            "frame": {"duration": 0, "redraw": True},
                            "transition": {"duration": 0}
                        }
                    ]
                )
                for j in range(len(frames))
            ]
        )
    ]
)

st.plotly_chart(
    animation_fig,
    use_container_width=True,
    key="shm_2d_animation"
)

st.caption(
    "Animation uses the RK4 numerical solution. "
    "The yellow dashed line indicates equilibrium."
)


# =====================================================
# 10. ADVANCED 3D SPRING-MASS LABORATORY
# =====================================================

st.divider()
st.header("🌐 Interactive 3D Spring-Mass Laboratory")
st.write(
    "Explore a three-dimensional spring-mass model. "
    "Drag to orbit, scroll to zoom, and use the animation controls "
    "to compare the analytical solution with RK4."
)

# Display coordinates are deliberately separate from the physical
# displacement values used in the numerical calculations.
display_scale_3d = 2.0
mass_size_3d = 0.42
spring_radius_3d = 0.115
spring_tube_radius_3d = 0.022
spring_turns_3d = 10
rk4_lane_3d = -0.42
theory_lane_3d = 0.42

# Keep the 3D animation responsive even for small integration steps.
indices_3d = np.unique(
    np.linspace(0, steps, min(90, steps + 1), dtype=int)
)

# Fixed mesh topology: each spring has the same number of vertices
# and faces in every frame, allowing Plotly to animate its geometry.
spring_path_points_3d = 100
spring_ring_points_3d = 8
spring_theta_3d = np.linspace(
    0, 2 * np.pi * spring_turns_3d,
    spring_path_points_3d
)
spring_phi_3d = np.linspace(
    0, 2 * np.pi,
    spring_ring_points_3d,
    endpoint=False
)

spring_face_i_3d = []
spring_face_j_3d = []
spring_face_k_3d = []

for p in range(spring_path_points_3d - 1):
    for q in range(spring_ring_points_3d):
        q_next = (q + 1) % spring_ring_points_3d
        a = p * spring_ring_points_3d + q
        b = p * spring_ring_points_3d + q_next
        c = (p + 1) * spring_ring_points_3d + q
        d = (p + 1) * spring_ring_points_3d + q_next

        spring_face_i_3d.extend([a, b])
        spring_face_j_3d.extend([c, c])
        spring_face_k_3d.extend([d, d])


def spring_mesh_3d(mass_position, lane):
    """Return vertices for a tubular helical spring along the X axis."""
    start_x = wall_x + 0.12
    end_x = mass_position - mass_size_3d / 2
    center_x = np.linspace(start_x, end_x, spring_path_points_3d)

    theta = spring_theta_3d
    center_y = lane + spring_radius_3d * np.cos(theta)
    center_z = spring_radius_3d * np.sin(theta)

    # Tangent to the helix, normalized at every sample.
    dx_dtheta = (end_x - start_x) / (2 * np.pi * spring_turns_3d)
    tangent = np.column_stack((
        np.full(spring_path_points_3d, dx_dtheta),
        -spring_radius_3d * np.sin(theta),
        spring_radius_3d * np.cos(theta)
    ))
    tangent /= np.maximum(
        np.linalg.norm(tangent, axis=1)[:, None],
        1e-12
    )

    # A stable perpendicular basis around the helix.
    normal = np.column_stack((
        np.zeros(spring_path_points_3d),
        np.cos(theta),
        np.sin(theta)
    ))
    binormal = np.cross(tangent, normal)
    binormal /= np.maximum(
        np.linalg.norm(binormal, axis=1)[:, None],
        1e-12
    )

    cos_phi = np.cos(spring_phi_3d)[None, :, None]
    sin_phi = np.sin(spring_phi_3d)[None, :, None]

    normal_3d = normal[:, None, :]
    binormal_3d = binormal[:, None, :]

    ring_offsets = spring_tube_radius_3d * (
        cos_phi * normal_3d + sin_phi * binormal_3d
    )

    centers = np.column_stack((center_x, center_y, center_z))
    vertices = centers[:, None, :] + ring_offsets

    return (
        vertices[:, :, 0].reshape(-1),
        vertices[:, :, 1].reshape(-1),
        vertices[:, :, 2].reshape(-1)
    )


def cube_mesh_3d(center_x, center_y, center_z=0.0):
    """Return a closed cuboid mesh centered at the supplied position."""
    h = mass_size_3d / 2
    vertices = np.array([
        [center_x-h, center_y-h, center_z-h],
        [center_x+h, center_y-h, center_z-h],
        [center_x+h, center_y+h, center_z-h],
        [center_x-h, center_y+h, center_z-h],
        [center_x-h, center_y-h, center_z+h],
        [center_x+h, center_y-h, center_z+h],
        [center_x+h, center_y+h, center_z+h],
        [center_x-h, center_y+h, center_z+h]
    ])

    return (
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        [0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3],
        [2, 3, 5, 6, 1, 5, 2, 6, 3, 7, 0, 4],
        [1, 2, 6, 7, 5, 4, 6, 5, 7, 6, 4, 7]
    )


def vector_cone_3d(position, lane, value, scale, color, name):
    """Create an X-axis cone vector; its sign indicates direction."""
    vector_length = float(np.clip(value * scale, -0.85, 0.85))

    return go.Cone(
        x=[position],
        y=[lane],
        z=[0.0],
        u=[vector_length],
        v=[0.0],
        w=[0.0],
        anchor="tail",
        sizemode="absolute",
        sizeref=0.16,
        colorscale=[[0, color], [1, color]],
        showscale=False,
        name=name,
        hovertemplate=f"{name}<extra></extra>"
    )


def spring_trace_3d(position, lane, color, name):
    sx, sy, sz = spring_mesh_3d(position, lane)
    return go.Mesh3d(
        x=sx,
        y=sy,
        z=sz,
        i=spring_face_i_3d,
        j=spring_face_j_3d,
        k=spring_face_k_3d,
        color=color,
        flatshading=False,
        lighting=dict(
            ambient=0.38,
            diffuse=0.9,
            specular=1.0,
            roughness=0.22,
            fresnel=0.35
        ),
        lightposition=dict(x=100, y=150, z=200),
        name=name,
        hoverinfo="skip"
    )


def mass_trace_3d(position, lane, color, name):
    mx, my, mz, mi, mj, mk = cube_mesh_3d(position, lane)
    return go.Mesh3d(
        x=mx,
        y=my,
        z=mz,
        i=mi,
        j=mj,
        k=mk,
        color=color,
        flatshading=False,
        lighting=dict(
            ambient=0.4,
            diffuse=0.9,
            specular=1.0,
            roughness=0.2
        ),
        name=name,
        hoverinfo="skip"
    )


first_3d_index = indices_3d[0]
first_rk4_position_3d = (
    equilibrium_x + display_scale_3d * x_rk4[first_3d_index]
)
first_theory_position_3d = (
    equilibrium_x + display_scale_3d * x_theory[first_3d_index]
)

# Trace order is important: animation frames update traces 0, 1, 2, 3, 7, 8.
animation_3d = go.Figure()

animation_3d.add_trace(
    spring_trace_3d(
        first_rk4_position_3d,
        rk4_lane_3d,
        "#9CC7E8",
        "RK4 metallic spring"
    )
)
animation_3d.add_trace(
    mass_trace_3d(
        first_rk4_position_3d,
        rk4_lane_3d,
        "#A78BFA",
        "RK4 mass"
    )
)
animation_3d.add_trace(
    spring_trace_3d(
        first_theory_position_3d,
        theory_lane_3d,
        "#F59E0B",
        "Theory metallic spring"
    )
)
animation_3d.add_trace(
    mass_trace_3d(
        first_theory_position_3d,
        theory_lane_3d,
        "#22C55E",
        "Theory mass"
    )
)

# Fixed support wall.
wall_vertices_3d = np.array([
    [wall_x-0.09, -1.0, -0.55],
    [wall_x+0.09, -1.0, -0.55],
    [wall_x+0.09, 1.0, -0.55],
    [wall_x-0.09, 1.0, -0.55],
    [wall_x-0.09, -1.0, 0.55],
    [wall_x+0.09, -1.0, 0.55],
    [wall_x+0.09, 1.0, 0.55],
    [wall_x-0.09, 1.0, 0.55]
])

animation_3d.add_trace(
    go.Mesh3d(
        x=wall_vertices_3d[:, 0],
        y=wall_vertices_3d[:, 1],
        z=wall_vertices_3d[:, 2],
        i=[0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3],
        j=[2, 3, 5, 6, 1, 5, 2, 6, 3, 7, 0, 4],
        k=[1, 2, 6, 7, 5, 4, 6, 5, 7, 6, 4, 7],
        color="#64748B",
        flatshading=False,
        name="Fixed support",
        hoverinfo="skip"
    )
)

# Equilibrium markers for both comparison lanes.
for lane, color, label in [
    (rk4_lane_3d, "#38BDF8", "RK4 equilibrium"),
    (theory_lane_3d, "#F59E0B", "Theory equilibrium")
]:
    animation_3d.add_trace(
        go.Scatter3d(
            x=[equilibrium_x, equilibrium_x],
            y=[lane, lane],
            z=[-0.55, 0.55],
            mode="lines",
            line=dict(color=color, width=5, dash="dash"),
            name=label,
            hoverinfo="skip"
        )
    )

# Live vectors: velocity and acceleration for RK4.
animation_3d.add_trace(
    vector_cone_3d(
        first_rk4_position_3d,
        rk4_lane_3d - 0.19,
        v_rk4[first_3d_index],
        0.45,
        "#38BDF8",
        "RK4 velocity vector"
    )
)
animation_3d.add_trace(
    vector_cone_3d(
        first_rk4_position_3d,
        rk4_lane_3d + 0.19,
        a_rk4[first_3d_index],
        0.10,
        "#F97316",
        "RK4 acceleration vector"
    )
)

# Lane labels.
animation_3d.add_trace(
    go.Scatter3d(
        x=[wall_x - 0.05, wall_x - 0.05],
        y=[rk4_lane_3d, theory_lane_3d],
        z=[0.72, 0.72],
        mode="text",
        text=["RK4 NUMERICAL", "THEORETICAL"],
        textfont=dict(size=12, color="#E2E8F0"),
        showlegend=False,
        hoverinfo="skip"
    )
)

frames_3d = []

for frame_number, index in enumerate(indices_3d):
    rk4_position = (
        equilibrium_x + display_scale_3d * x_rk4[index]
    )
    theory_position = (
        equilibrium_x + display_scale_3d * x_theory[index]
    )

    rk4_sx, rk4_sy, rk4_sz = spring_mesh_3d(
        rk4_position, rk4_lane_3d
    )
    theory_sx, theory_sy, theory_sz = spring_mesh_3d(
        theory_position, theory_lane_3d
    )

    rk4_mx, rk4_my, rk4_mz, _, _, _ = cube_mesh_3d(
        rk4_position, rk4_lane_3d
    )
    theory_mx, theory_my, theory_mz, _, _, _ = cube_mesh_3d(
        theory_position, theory_lane_3d
    )

    velocity_length = float(
        np.clip(v_rk4[index] * 0.45, -0.85, 0.85)
    )
    acceleration_length = float(
        np.clip(a_rk4[index] * 0.10, -0.85, 0.85)
    )

    title_text_3d = (
        f"t = {time[index]:.3f} s"
        f"  |  x = {x_rk4[index]:.4f} m"
        f"  |  v = {v_rk4[index]:.4f} m/s"
        f"  |  a = {a_rk4[index]:.4f} m/s²"
    )

    frames_3d.append(
        go.Frame(
            name=str(frame_number),
            data=[
                go.Mesh3d(x=rk4_sx, y=rk4_sy, z=rk4_sz),
                go.Mesh3d(x=rk4_mx, y=rk4_my, z=rk4_mz),
                go.Mesh3d(x=theory_sx, y=theory_sy, z=theory_sz),
                go.Mesh3d(x=theory_mx, y=theory_my, z=theory_mz),
                go.Cone(
                    x=[rk4_position],
                    y=[rk4_lane_3d - 0.19],
                    z=[0.0],
                    u=[velocity_length],
                    v=[0.0],
                    w=[0.0]
                ),
                go.Cone(
                    x=[rk4_position],
                    y=[rk4_lane_3d + 0.19],
                    z=[0.0],
                    u=[acceleration_length],
                    v=[0.0],
                    w=[0.0]
                )
            ],
            traces=[0, 1, 2, 3, 7, 8],
            layout=go.Layout(title_text=title_text_3d)
        )
    )

animation_3d.frames = frames_3d

animation_3d.update_layout(
    template="plotly_dark",
    height=780,
    margin=dict(l=0, r=0, t=110, b=25),
    title=dict(
        text=(
            f"3D SHM | t = {time[first_3d_index]:.3f} s"
            f" | x = {x_rk4[first_3d_index]:.4f} m"
        ),
        x=0.02
    ),
    scene=dict(
        xaxis=dict(
            title="X — Oscillation direction",
            range=[wall_x - 0.3, equilibrium_x + 2.0],
            backgroundcolor="#101820",
            gridcolor="#475569",
            zerolinecolor="#64748B"
        ),
        yaxis=dict(
            title="Comparison lane",
            range=[-1.25, 1.25],
            backgroundcolor="#101820",
            gridcolor="#475569"
        ),
        zaxis=dict(
            title="Z — Spring coil depth",
            range=[-0.9, 0.9],
            backgroundcolor="#101820",
            gridcolor="#475569"
        ),
        aspectmode="manual",
        aspectratio=dict(x=2.5, y=1.3, z=1.0),
        camera=dict(
            eye=dict(x=1.8, y=1.8, z=1.25),
            up=dict(x=0, y=0, z=1),
            projection=dict(type="perspective")
        ),
        dragmode="orbit"
    ),
    legend=dict(
        orientation="h",
        y=-0.08
    ),
    hovermode=False,
    uirevision="shm-3d-camera",
    updatemenus=[
        dict(
            type="buttons",
            direction="left",
            x=0.02,
            y=1.16,
            buttons=[
                dict(
                    label="▶ PLAY",
                    method="animate",
                    args=[
                        None,
                        {
                            "frame": {"duration": 55, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 0}
                        }
                    ]
                ),
                dict(
                    label="⏸ PAUSE",
                    method="animate",
                    args=[
                        [None],
                        {
                            "mode": "immediate",
                            "frame": {"duration": 0, "redraw": False},
                            "transition": {"duration": 0}
                        }
                    ]
                ),
                dict(
                    label="↺ RESET",
                    method="animate",
                    args=[
                        ["0"],
                        {
                            "mode": "immediate",
                            "frame": {"duration": 0, "redraw": True},
                            "transition": {"duration": 0}
                        }
                    ]
                )
            ]
        )
    ],
    sliders=[
        dict(
            active=0,
            currentvalue=dict(prefix="3D frame: "),
            steps=[
                dict(
                    label=f"{time[index]:.2f}s",
                    method="animate",
                    args=[
                        [str(j)],
                        {
                            "mode": "immediate",
                            "frame": {"duration": 0, "redraw": True},
                            "transition": {"duration": 0}
                        }
                    ]
                )
                for j, index in enumerate(indices_3d)
            ]
        )
    ]
)

st.plotly_chart(
    animation_3d,
    use_container_width=True,
    key="shm_3d_engineering_lab",
    config={
        "scrollZoom": True,
        "displaylogo": False,
        "responsive": True
    }
)

# Static numerical inspection controls are separate from the Plotly
# animation timeline, so the displayed values are always unambiguous.
st.subheader("3D Laboratory — Numerical Readout")
inspection_frame_3d = st.slider(
    "Inspect a calculated time point",
    min_value=0,
    max_value=len(indices_3d) - 1,
    value=0,
    key="shm_3d_inspection"
)
inspection_index_3d = indices_3d[inspection_frame_3d]

readout_cols_3d = st.columns(4)
readout_cols_3d[0].metric(
    "Time",
    f"{time[inspection_index_3d]:.3f} s"
)
readout_cols_3d[1].metric(
    "Displacement",
    f"{x_rk4[inspection_index_3d]:.5f} m"
)
readout_cols_3d[2].metric(
    "Velocity",
    f"{v_rk4[inspection_index_3d]:.5f} m/s"
)
readout_cols_3d[3].metric(
    "Acceleration",
    f"{a_rk4[inspection_index_3d]:.5f} m/s²"
)

st.caption(
    "Blue lane: RK4 numerical solution. Gold spring / green mass: "
    "analytical theoretical solution. Blue arrow: velocity. "
    "Orange arrow: acceleration. Use the mouse to orbit and zoom."
)


# =====================================================
# 11. NUMERICAL ACCURACY LABORATORY
# =====================================================


# =====================================================

st.divider()
st.header("🔬 Numerical Accuracy Laboratory")

st.write(
    "Compare the Euler and RK4 numerical methods "
    "against the theoretical SHM solution."
)

# Error functions
def calculate_rmse(actual, predicted):
    return np.sqrt(
        np.mean((actual - predicted) ** 2)
    )


def calculate_max_error(actual, predicted):
    return np.max(
        np.abs(actual - predicted)
    )


# Displacement errors
euler_x_rmse = calculate_rmse(x_theory, x_euler)
rk4_x_rmse = calculate_rmse(x_theory, x_rk4)

euler_x_max = calculate_max_error(x_theory, x_euler)
rk4_x_max = calculate_max_error(x_theory, x_rk4)

# Velocity errors
euler_v_rmse = calculate_rmse(v_theory, v_euler)
rk4_v_rmse = calculate_rmse(v_theory, v_rk4)

euler_v_max = calculate_max_error(v_theory, v_euler)
rk4_v_max = calculate_max_error(v_theory, v_rk4)

# Acceleration errors
euler_a_rmse = calculate_rmse(a_theory, a_euler)
rk4_a_rmse = calculate_rmse(a_theory, a_rk4)

euler_a_max = calculate_max_error(a_theory, a_euler)
rk4_a_max = calculate_max_error(a_theory, a_rk4)

# Energy deviations
euler_energy_error = np.max(
    np.abs(euler_energy - initial_energy)
)

rk4_energy_error = np.max(
    np.abs(rk4_energy - initial_energy)
)

# =====================================================
# 12. ACCURACY METRIC CARDS
# =====================================================

st.subheader("Method Accuracy")

col1, col2 = st.columns(2)

with col1:

    st.markdown("### Euler Method")

    st.metric(
        "Displacement RMSE",
        f"{euler_x_rmse:.8e} m"
    )

    st.metric(
        "Velocity RMSE",
        f"{euler_v_rmse:.8e} m/s"
    )

    st.metric(
        "Acceleration RMSE",
        f"{euler_a_rmse:.8e} m/s²"
    )

    st.metric(
        "Maximum Displacement Error",
        f"{euler_x_max:.8e} m"
    )

    st.metric(
        "Energy Deviation",
        f"{euler_energy_error:.8e} J"
    )


with col2:

    st.markdown("### RK4 Method")

    st.metric(
        "Displacement RMSE",
        f"{rk4_x_rmse:.8e} m"
    )

    st.metric(
        "Velocity RMSE",
        f"{rk4_v_rmse:.8e} m/s"
    )

    st.metric(
        "Acceleration RMSE",
        f"{rk4_a_rmse:.8e} m/s²"
    )

    st.metric(
        "Maximum Displacement Error",
        f"{rk4_x_max:.8e} m"
    )

    st.metric(
        "Energy Deviation",
        f"{rk4_energy_error:.8e} J"
    )

# =====================================================
# 13. DETAILED COMPARISON TABLE
# =====================================================

st.subheader("Detailed Error Comparison")

comparison_data = {
    "Measurement": [
        "Displacement RMSE",
        "Velocity RMSE",
        "Acceleration RMSE",
        "Maximum Displacement Error",
        "Maximum Velocity Error",
        "Maximum Acceleration Error",
        "Maximum Energy Deviation"
    ],

    "Euler": [
        f"{euler_x_rmse:.8e} m",
        f"{euler_v_rmse:.8e} m/s",
        f"{euler_a_rmse:.8e} m/s²",
        f"{euler_x_max:.8e} m",
        f"{euler_v_max:.8e} m/s",
        f"{euler_a_max:.8e} m/s²",
        f"{euler_energy_error:.8e} J"
    ],

    "RK4": [
        f"{rk4_x_rmse:.8e} m",
        f"{rk4_v_rmse:.8e} m/s",
        f"{rk4_a_rmse:.8e} m/s²",
        f"{rk4_x_max:.8e} m",
        f"{rk4_v_max:.8e} m/s",
        f"{rk4_a_max:.8e} m/s²",
        f"{rk4_energy_error:.8e} J"
    ]
}

st.dataframe(
    comparison_data,
    use_container_width=True,
    hide_index=True
)

# =====================================================
# 14. ERROR VS TIME GRAPH
# =====================================================

st.subheader("Absolute Displacement Error vs Time")

error_fig = go.Figure()

error_fig.add_trace(
    go.Scatter(
        x=time,
        y=np.abs(x_theory - x_euler),
        name="Euler Error",
        line=dict(color="#F97316", width=2)
    )
)

error_fig.add_trace(
    go.Scatter(
        x=time,
        y=np.abs(x_theory - x_rk4),
        name="RK4 Error",
        line=dict(color="#38BDF8", width=2)
    )
)

error_fig.update_layout(
    template="plotly_dark",
    height=450,
    xaxis_title="Time (seconds)",
    yaxis_title="Absolute Error (m)",
    hovermode="x unified"
)

st.plotly_chart(
    error_fig,
    use_container_width=True
)

# =====================================================
# 15. SCIENTIFIC CONCLUSION
# =====================================================

st.subheader("Scientific Conclusion")

if rk4_x_rmse < euler_x_rmse:

    st.success(
        "RK4 produced a lower displacement RMSE "
        "than Euler for the current simulation settings."
    )

else:

    st.info(
        "Euler produced a lower displacement RMSE "
        "than RK4 for the current simulation settings."
    )

st.write(
    f"Euler displacement RMSE: {euler_x_rmse:.8e} m"
)

st.write(
    f"RK4 displacement RMSE: {rk4_x_rmse:.8e} m"
)

st.write(
    f"Euler energy deviation: {euler_energy_error:.8e} J"
)

st.write(
    f"RK4 energy deviation: {rk4_energy_error:.8e} J"
)

# =====================================================
# 16. PROJECT FOOTER
# =====================================================

st.divider()

st.caption(
    "SHM Physics Mini Project | "
    "Interactive Spring-Mass Simulation | "
    "Euler and Fourth-Order Runge-Kutta Methods"
)
