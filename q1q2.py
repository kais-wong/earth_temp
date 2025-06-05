import numpy as np
import matplotlib.pyplot as plt
import time

# === Constants ===
sigma = 5.67e-8          # Stefan-Boltzmann constant (W/m^2/K^4)
alpha = 0.3              # Albedo
S_0 = 1367               # Solar constant (W/m^2)
epsilon_LW = 0.61        # Longwave emissivity
rho = 1000               # Density of water (kg/m^3)
cp = 4218                # Specific heat capacity of water (J/kg/K)
h_ml = 70                # Mixed layer depth (m)
C_eff = rho * cp * h_ml  # Effective heat capacity (J/m^2/K)

# === Simulation Functions ===
def albedo(temp):
    """Determine the albedo for a given temp"""
    temp = temp - 273.15  # Temperature in °C
    if temp >= 10 :
      return 0.3
    elif temp <= -10 :
      return 0.7
    else :
      return -0.02 * temp + 0.5

def euler_method(T_init, steps, dt, S, albedo_mode="fixed"):
    """Evaluate climate model using Euler Integration"""
    T = np.zeros(steps)
    T[0] = T_init

    for i in range(steps - 1):
        if albedo_mode == "fixed":
          alpha_i = alpha
        elif albedo_mode == "variable":
          alpha_i = albedo(T[i])
        else:
          raise ValueError("Invalid albedo_mode. Choose 'fixed' or 'variable'.")

        dT = 1 / C_eff * ((1 - alpha_i) * S / 4 - epsilon_LW * sigma * T[i]**4)
        T[i + 1] = T[i] + dt * dT

    return T

def run_equilibrium_sequence(S_vals, T_init, dt, years, mode='fixed'):
    """Returns an array with equilibrium temperatures for variable S based on pervious temperatures"""
    T_eq = np.zeros_like(S_vals)
    T_eq[0] = T_init
    steps = int(years * 365 * 24 * 3600 / dt)
    for i, S in enumerate(S_vals):
        T_eq[i] = euler_method(T_eq[i - 1], steps, dt, S, albedo_mode=mode)[-1]
    return T_eq

# === Plotting Functions ===
def plot_temperature_time_series(time, T, title, label=None):
    """Plots temperature vs time series, for Q1(a) and Q1(b)"""
    plt.plot(time, T - 273.15, label=label)
    plt.xlabel('Time (years)')
    plt.ylabel('Surface Temperature $T_s$ (°C)')
    plt.title(title)
    plt.grid(True)
    if label:
        plt.legend()

def plot_equilibrium_vs_solar(S_vals, T_eqs, title, label=None, color=None):
    """Plots equilibrium temperature vs solar constant factor for Q1(c), Q2(b) and Q2(c)"""
    plt.plot(S_vals / S_0, T_eqs - 273.15, 'o-', label=label, color=color)
    plt.xlabel('Solar Constant Factor ($S/S_0$)')
    plt.ylabel('Equilibrium Temperature $T_s$ (°C)')
    plt.title(title)
    plt.grid(True)
    if label:
        plt.legend()

# === Main Workflow ===
def main():
    # Start timing for performance profiling
    start_time = time.time()

    # Time settings
    years = 100
    T_init = 273.15              # Intial temp of 0°C in Kelvin
    dt_daily = 24 * 3600         # One day in seconds
    dt_yearly = 365 * dt_daily   # One year in seconds

    # === Q1(a): Daily Time Steps ===

    print("Solving Question 1, Part(a) ...")

    # For plotting
    steps_daily = int(years * 365)                     # Number of days, total number of time steps
    time_daily = np.linspace(0, years, steps_daily)    # For plotting each day [0, 50 years]
    T_daily = euler_method(T_init, steps_daily, dt_daily, S_0)
    plt.figure(figsize=(9,6))
    plot_temperature_time_series(time_daily, T_daily, "Q1(a): Surface Temperature over 50 Years (daily time steps)")
    plt.show()

    # === Q1(b): Yearly Time Steps ===

    print("Solving Question 1, Part(b) ...")
    t2 = time.time()

    # For plotting
    steps_yearly = int(years)                           # Number of years, total number of time steps
    time_yearly = np.linspace(0, years, steps_yearly)   # For plotting each year [0, 50 years]
    T_yearly = euler_method(T_init, steps_yearly, dt_yearly, S_0)
    plt.figure(figsize=(9,6))
    plot_temperature_time_series(time_yearly, T_yearly, "Q1(b): Surface Temperature over 50 Years (yearly time steps)")
    plt.show()

    print(f"Part(b) completed in {time.time() - t2:.2f} seconds")

    # === Compare Daily vs Yearly Time Steps ===

    plt.figure(figsize=(9,6))
    plot_temperature_time_series(time_daily, T_daily, "Q1(a) vs Q1(b): Daily vs. Yearly Time Steps Comparison", label='Daily Steps') # Plot daily data
    plot_temperature_time_series(time_yearly, T_yearly, "Q1(a) vs Q1(b): Daily vs. Yearly Time Steps Comparison", label='Yearly Steps') # Plot yearly data on the same figure
    plt.legend() # Add a legend to distinguish the lines
    plt.show() # Display the combined plot

    # === Question 1, Part(c): Equilibrium Temperature for Variable S ===

    print("Solving Question 1, Part(c) ...")

    # Intial Conditions (daily time steps)
    S_values = np.array([2 * S_0 * 0.9**i for i in range(16)])  # From 2S0 to 0.4S0, length = base_0.9_log(0.4) = 15.28
    T_eq = np.zeros(16)
    S_dt = dt_daily
    S_steps = steps_daily

    for i in range(len(S_values)):

      # Store the final temperature into T_eq and record S value
      T_eq[i] = euler_method(T_init, S_steps, S_dt, S_values[i])[-1]


    # For plotting
    plt.figure(figsize=(9,6))
    plot_equilibrium_vs_solar(S_values, T_eq, title="Q1(c): Equilibrium Temperature vs Solar Constant")
    plt.show()

    # === Question 2, Part(b): Equilibrium Temperature for Variable S ===

    print("Solving Question 2, Part(b) and Part (c )...")

    S_values2 = np.array([2 * S_0 * 0.9**i for i in range(16)])  # From 2S0 to 0.4S0, length = base_0.9_log(0.4) = 15.28
    T_eq2 = run_equilibrium_sequence(S_values2, 273.15, dt_daily, years, mode='variable')

    # For plotting
    plt.figure(figsize=(9,6))
    plot_equilibrium_vs_solar(S_values2, T_eq2, title="Q2(b) & Q2(c): Hysteresis", label="Cooling")

    # === Question 2, Part(c): Equilibrium Temperature for Variable S ===

    # S_vals3 = np.array([S_values2[-1] * 1.1**i for i in range(17)])
    S_values3 = np.array(S_values[::-1])      # Use same S_vals as Q2(b) but in reverse
    T_eq3 = run_equilibrium_sequence(S_values3, T_eq2[-1], dt_daily, years, mode='variable')

    # For plotting
    plot_equilibrium_vs_solar(S_values3, T_eq3, title="Q2(b) & Q2(c): Hysteresis", label="Warming")
    plt.show()

    # Report total runtime
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
