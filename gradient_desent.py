import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# population
u = np.array([2, 5, 14, 34, 56, 94, 189, 266, 330, 416, 507, 580, 610, 
                       513, 593, 557, 560, 522, 565, 517, 500, 585, 500, 495, 525, 510])

# the derivative of population
udot = np.diff(u, append=u[-1])

# predefined constant
period=len(u)
epsilon=0.05
lr=0.01 #learning rate
iter_limit=100

# initial parameters
params=np.array([[2,1,0.5,1]]) # random initalization

def SSE(a,b,h,k,udot):
    return np.sum((udot-(a*u**h-b*u**k))**2)

def gradient_desent(params,udot):
    a,b,h,k=params[-1]
    SSE_prev=SSE(a,b,h,k,udot)
    for _ in range(iter_limit):
        SSE_a_plus = SSE(a + epsilon, b, h, k, udot)
        SSE_b_plus = SSE(a, b + epsilon, h, k, udot)
        SSE_h_plus = SSE(a, b, h + epsilon, k, udot)
        SSE_k_plus = SSE(a, b, h, k + epsilon, udot)

        dSSE_da = (SSE_a_plus - SSE_prev) / epsilon
        dSSE_db = (SSE_b_plus - SSE_prev) / epsilon
        dSSE_dh = (SSE_h_plus - SSE_prev) / epsilon
        dSSE_dk = (SSE_k_plus - SSE_prev) / epsilon

        total=dSSE_da+dSSE_db+dSSE_dh+dSSE_dk

        a_next = a - lr * dSSE_da/total
        b_next = b - lr * dSSE_db/total
        h_next = h - lr * dSSE_dh/total
        k_next = k - lr * dSSE_dk/total

        SSE_curr = SSE(a_next, b_next, h_next, k_next, udot)

        if SSE_prev-SSE_curr<0:
            break
        
        a, b, h, k = a_next, b_next, h_next, k_next
        SSE_prev = SSE_curr

        params=np.concatenate((params,np.array([[a,b,h,k]])),axis=0)
    
    return params

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

def plot_OES(params,u):
    OES=[OES_of_sol(solve_ODE(a,b,h,k),u) for a,b,h,k in params]

    iteration=np.arange(len(params))
    plt.figure(figsize=(10,7))
    plt.plot(iteration,OES,marker='o', linestyle='-')

    plt.xlabel("Iteration")
    plt.ylabel("OES")
    plt.title("OES during gradient desent")
    plt.show()
    
def plot_SSE(params, udot):
    SSE_values = [SSE(a, b, h, k, udot) for a, b, h, k in params]
    
    iteration = np.arange(len(params))
    plt.figure(figsize=(10, 7))
    plt.plot(iteration, SSE_values, marker='o', linestyle='-')

    plt.xlabel("Iteration")
    plt.ylabel("SSE")
    plt.title("SSE during Gradient Descent")
    plt.show()    

def plot_improvement(params, u):
    initial_params = params[0]
    optimized_params = params[-1]

    sol_initial = solve_ODE(*initial_params)
    sol_optimized = solve_ODE(*optimized_params)

    t_values = np.arange(len(u))

    plt.figure(figsize=(10, 7))
    
    plt.plot(t_values, u, 'bo-', label="Actual Data")

    plt.plot(t_values, sol_initial.y[0], linestyle="dashed", color='red', 
             label=f"Initial Solution (a={initial_params[0]:.3f}, b={initial_params[1]:.3f}, h={initial_params[2]:.3f}, k={initial_params[3]:.3f})")

    plt.plot(t_values, sol_optimized.y[0], linestyle="-", color='red', 
             label=f"Optimized Solution (a={optimized_params[0]:.3f}, b={optimized_params[1]:.3f}, h={optimized_params[2]:.3f}, k={optimized_params[3]:.3f})")

    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.title("Comparison of Initial and Optimized ODE Solutions")
    plt.legend(loc="lower right")
    plt.show()


params=gradient_desent(params,udot)
plot_OES(params,u)
plot_SSE(params,udot)
plot_improvement(params,u)



