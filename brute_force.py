import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# population
u = np.array([2, 5, 14, 34, 56, 94, 189, 266, 330, 416, 507, 580, 610, 
                       513, 593, 557, 560, 522, 565, 517, 500, 585, 500, 495, 525, 510])

# the derivative of population
udot = np.diff(u, append=u[-1])

# set the range of h and k
h_values = np.arange(0, 3, 0.1)
k_values = np.arange(0, 3, 0.1)

# predefined constant
period=len(u)

def OLS_regression(h,k,u,udot):
    u_h=u**h
    u_k=u**k
    X=np.vstack([u_h,-u_k]).T
    a,b=np.linalg.inv(X.T@X)@X.T@udot
    return a,b

def solve_ODE(a,b,h,k):
    # Define ODE
    def xdot(t, x):
        return a * x**h - b * x**k  # du/dt = a*u^h - b*u^k
    
    t_span = (0, period-1)
    t_eval = np.linspace(0, period-1, period)

    sol = solve_ivp(xdot, t_span, (u[0],), t_eval=t_eval)
    return sol

def OES_of_sol(sol,u):
    u_sol=sol.y[0]
    oes=np.sum((u-u_sol)**2)
    return oes

def OES_surface(h_values, k_values, u, udot, epsilon=0.05):
    H, K = np.meshgrid(h_values, k_values)
    OES_values = np.full_like(H, np.nan)

    for i in range(len(h_values)):
        for j in range(len(k_values)):
            h, k = H[j, i], K[j, i]

            if abs(h - k) < epsilon:
                continue

            a, b = OLS_regression(h, k, u, udot)
            sol = solve_ODE(a, b, h, k)
            OES_values[j, i] = OES_of_sol(sol, u)

    return H, K, OES_values


def OES_surface_plot(H, K, OES_values):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(H, K, OES_values, cmap='viridis', edgecolor='none')

    ax.set_xlabel("h values")
    ax.set_ylabel("k values")
    ax.set_zlabel("OES")
    ax.set_title("OES Surface Plot")

    plt.show()


def OES_surface_best(H, K, OES_values, top_n=5):
    valid_indices = ~np.isnan(OES_values)
    sorted_indices = np.argsort(OES_values[valid_indices])

    best_hk_pairs = []
    valid_H, valid_K, valid_OES = H[valid_indices], K[valid_indices], OES_values[valid_indices]

    for i in range(min(top_n, len(sorted_indices))):
        idx = sorted_indices[i]
        best_hk_pairs.append((valid_H[idx], valid_K[idx], valid_OES[idx]))

    return best_hk_pairs


def Plot_best_sol(best_hk_pairs, u):
    t = np.arange(period)
    plt.figure(figsize=(10, 6))
    plt.plot(t, u, 'bo-', label="Actual Data")

    for h, k, oes in best_hk_pairs:
        a, b = OLS_regression(h, k, u, udot)
        sol = solve_ODE(a, b, h, k)
        plt.plot(sol.t, sol.y[0], label=f"h={h:.2f}, k={k:.2f}, OES={oes:.2f}")

    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.title("Best Solutions vs Actual Data")
    plt.legend()
    plt.grid(True)
    plt.show()


H, K, OES_values = OES_surface(h_values, k_values, u, udot)

OES_surface_plot(H, K, OES_values)

best_hk_pairs = OES_surface_best(H, K, OES_values, top_n=5)

Plot_best_sol(best_hk_pairs, u)
    
