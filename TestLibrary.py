import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

#heat_equation_rod()
#uncomment this line to test graph

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
    
 