import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def plot_projectile_motion(v0, angle, air_res=0.0, mass=1.0, dt=0.1, brownian_strength=0.1):
    """
    Simulates and plots projectile motion with air resistance.

    Args:
        v0 (float): Initial velocity (m/s)
        angle (float): Launch angle (degrees)
        air_res (float): Air resistance coefficient (kg/s)
        mass (float): Mass of projectile (kg)
        dt (float): Time step for simulation (s)
    """
    
    g = 9.81
    theta_rad = np.radians(angle)
    
    #Initial velocities
    vx = v0 * np.cos(angle)
    vy = v0 * np.sin(angle)
    
    # Initial position
    x, y = 0.0, 0.0
    
    t = 0.0
    data = []
    
    while y >= 0:
        # brownian motion
        bx = np.random.normal(0, brownian_strength)
        by = np.random.normal(0, brownian_strength)
        
        # Stores values each time step
        data.append([t, x+bx, y+by, vx, vy])
        
        v = np.sqrt(vx**2 + vy**2)
        ax = -air_res * vx / mass
        ay = -g - (air_res * vy / mass)
        
        # Euler integration
        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt
        t += dt
    
    df = pd.DataFrame(data, columns=["time", "x", "y", "vx", "vy"])
    
    # Plot trajectory
    plt.figure(figsize=(10, 5))
    plt.plot(df["x"], df["y"], label=f"v0={v0} m/s, θ={angle}°, k={air_res}")
    plt.title("Projectile Motion with Air Resistance and Brownian Motion")
    plt.xlabel("Distance (m)")
    plt.ylabel("Height (m)")
    plt.grid(True)
    plt.legend()
    plt.show()
    
    return df

#uncomment this line to test graph
#plot_projectile_motion(50, 45, air_res=0.1, mass=1.0, dt=0.1, brownian_strength=1.1)


#need to put in input paramaters
def heat_equation_rod():
    """
    Simulates the heat equation on a 1D rod using finite difference method.
    """
    # Parameters
    L = 10.0  # Length of the rod (m)
    T = 2.0   # Total time (s)
    Nx = 100  # Number of spatial points
    Nt = 200  # Number of time steps
    alpha = 0.01  # Thermal diffusivity (m^2/s)

    dx = L / (Nx - 1)  # Spatial step size
    dt = T / Nt       # Time step size

    # Stability condition
    if alpha * dt / dx**2 > 0.5:
        raise ValueError("Stability condition not met: alpha * dt / dx^2 <= 0.5")

    # Initial condition: u(x,0) = sin(pi*x/L)
    x = np.linspace(0, L, Nx)
    u = np.sin(np.pi * x / L)

    # Time-stepping loop
    for n in range(1, Nt):
        u_new = np.copy(u)
        for i in range(1, Nx - 1):
            u_new[i] = u[i] + alpha * dt / dx**2 * (u[i+1] - 2*u[i] + u[i-1])
        u = u_new
    
    df = pd.DataFrame({"x": x, "u": u})
    
    plt.figure(figsize=(10, 5))
    plt.plot(x, u, label=f"t={T} s")
    plt.title("Heat Equation on a 1D Rod")
    plt.xlabel("Position (m)")
    plt.ylabel("Temperature (°C)")
    plt.grid(True)
    plt.legend()
    plt.show()
    
    return df

def animate_heat_equation_rod():
    """
    Animates the evolution of the heat equation on a 1D rod using finite difference method.
    """
    # Parameters
    L = 10.0  # Length of the rod (m)
    T = 2.0   # Total time (s)
    Nx = 100  # Number of spatial points
    Nt = 200  # Number of time steps
    alpha = 0.01  # Thermal diffusivity (m^2/s)

    dx = L / (Nx - 1)  # Spatial step size
    dt = T / Nt       # Time step size

    # Stability condition
    if alpha * dt / dx**2 > 0.5:
        raise ValueError("Stability condition not met: alpha * dt / dx^2 <= 0.5")

    # Initial condition: u(x,0) = sin(pi*x/L)
    x = np.linspace(0, L, Nx)
    u = np.sin(np.pi * x / L)
    u_history = [u.copy()]

    # Time-stepping loop
    for n in range(1, Nt):
        u_new = np.copy(u)
        for i in range(1, Nx - 1):
            u_new[i] = u[i] + alpha * dt / dx**2 * (u[i+1] - 2*u[i] + u[i-1])
        u = u_new
        u_history.append(u.copy())

    u_history = np.array(u_history)

    # Animation
    fig, ax = plt.subplots(figsize=(10, 5))
    line, = ax.plot(x, u_history[0], color='blue')
    ax.set_title("Heat Equation on a 1D Rod (Animated)")
    ax.set_xlabel("Position (m)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_xlim(0, L)
    ax.set_ylim(np.min(u_history), np.max(u_history))
    cmap = plt.get_cmap('plasma')

    def update(frame):
        color = cmap(frame / Nt)
        line.set_ydata(u_history[frame])
        line.set_color(color)
        ax.set_title(f"Heat Equation on a 1D Rod (t={frame * dt:.2f} s)")
        return line,

    ani = animation.FuncAnimation(fig, update, frames=Nt, interval=50, blit=True, repeat=False)
    plt.show()
animate_heat_equation_rod()
#heat_equation_rod()
# uncomment this line to test graph

def one_dimensional_wave_equation():
    """
    Simulates the one-dimensional wave equation using finite difference method.
    """
    # Parameters
    L = 10.0  # Length of the string (m)
    T = 2.0   # Total time (s)
    Nx = 100  # Number of spatial points
    Nt = 200  # Number of time steps
    c = 1.0   # Wave speed (m/s)

    dx = L / (Nx - 1)  # Spatial step size
    dt = T / Nt       # Time step size

    # Stability condition
    if c * dt / dx > 1:
        raise ValueError("Stability condition not met: c * dt / dx <= 1")

    # Initial condition: u(x,0) = sin(pi*x/L)
    x = np.linspace(0, L, Nx)
    u = np.sin(np.pi * x / L)
    u_new = np.copy(u)
    u_prev = np.copy(u)

    u_prev[1:-1] += 0.5 * (c * dt / dx)**2 * (u[2:] - 2*u[1:-1] + u[:-2])
    
def visualize_eigenvectors(matrix):
    """
    Visualizes eigenvectors and eigenvalues of a given matrix.

    Args:
        matrix (np.ndarray): A matrix
    """
    
    vectors = np.array([[1,0],[0,1],[1,1],[1,-1]])
    
    transformed_vectors = np.dot(matrix, vectors.T).T
    
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    
    for vec in vectors:
        plt.quiver(0, 0, vec[0], vec[1], angles='xy', scale_units='xy', scale=1, color='blue', alpha=0.5, label="Original Vectors")
    
    # Plot transformed vectors
    for vec in transformed_vectors:
        plt.quiver(0, 0, vec[0], vec[1], angles='xy', scale_units='xy', scale=1, color='red', alpha=0.5, label="Transformed Vectors")
    
    # Plot eigenvectors
    for i in range(len(eigenvalues)):
        eig_vec = eigenvectors[:, i] * eigenvalues[i]
        plt.quiver(0, 0, eig_vec[0], eig_vec[1], angles='xy', scale_units='xy', scale=1, color='green', label=f"Eigenvector {i+1}")
    plt.title("Original Vectors")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    plt.legend(loc="upper left")
    plt.show()
    
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.title("Eigenvectors and Eigenvalues Visualization")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend(loc="upper left")
    plt.show()

# matrix = np.array([[2, 1],
#                    [1, 2]])
# visualize_eigenvectors(matrix)

def animate_eigenvector_transformation(matrix, steps=1000, interval=10):
    """
    Animates the evolution of vectors under repeated application of a matrix transformation.

    Args:
        matrix (np.ndarray): A 2x2 matrix.
        steps (int): Number of transformation steps.
        interval (int): Delay between frames in milliseconds.
    """
    vectors = np.array([[1, 0], [0, 1], [1, 1], [-1, 1]], dtype=float)
    original_vectors = vectors.copy()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Vectors Under Matrix Transformation")

    quivers = [ax.quiver(0, 0, vec[0], vec[1], angles='xy', scale_units='xy', scale=1, color='blue') for vec in original_vectors]

    def update(frame):
        nonlocal vectors
        # Normalize before transformation to prevent overflow
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vectors = vectors / norms
        vectors = np.dot(matrix, vectors.T).T
        for i, vec in enumerate(vectors):
            quivers[i].set_UVC(vec[0], vec[1])
        return quivers

    ani = animation.FuncAnimation(fig, update, frames=steps, interval=interval, blit=False, repeat=False)
    plt.show()
# Example usage:
# matrix = np.array([[2, 1],
#                    [1, 2]])
# animate_eigenvector_transformation(matrix, steps=1000, interval=10)