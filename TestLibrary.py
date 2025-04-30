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