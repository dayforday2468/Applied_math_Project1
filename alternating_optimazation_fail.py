import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# predefined constant
epsilon=0.05
lr=0.01 # learning rate
iter_limit=100
period=25



# population
u = np.array([2, 5, 14, 34, 56, 94, 189, 266, 330, 416, 507, 580, 610, 
                       513, 593, 557, 560, 522, 565, 517, 500, 585, 500, 495, 525, 510])

# the derivative of population
udot = np.diff(u, append=u[-1])

# parameters (a,b,h,k)
params=np.array([[np.nan,np.nan,1.5,2.5]])

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
    
    t_span = (0, period)
    t_eval = np.linspace(0, period, period+1)

    sol = solve_ivp(xdot, t_span, (u[0],), t_eval=t_eval)
    return sol

def OES_of_sol(sol,u):
    u_sol=sol.y[0]
    oes=np.sum((u-u_sol)**2)
    return oes

def SSE(a,b,h,k,udot):
    return np.sum((udot-(a*u**h-b*u**k))**2)


def gradient_descent(a, b, h, k, udot):
    h_cur = h
    k_cur = k
    SSE_prev = SSE(a,b,h,k,udot)
    
    for _ in range(iter_limit):
        SSE_h_plus = SSE(a,b,h+epsilon,k,udot)
        dSSE_dh = (SSE_h_plus - SSE_prev) / epsilon  # dSSE/dh

        SSE_k_plus = SSE(a,b,h,k+epsilon,udot)
        dSSE_dk = (SSE_k_plus - SSE_prev) / epsilon  # dSSE/dk

        total=dSSE_dh+dSSE_dk

        h_next = h_cur - lr * dSSE_dh/total
        k_next = k_cur - lr * dSSE_dk/total

        SSE_curr = SSE(a,b,h_next,k_next,udot)

        if SSE_prev - SSE_curr <0:
            break

        h_cur = h_next
        k_cur = k_next
        SSE_prev = SSE_curr

    return h_cur, k_cur

def alternating_optimazation(params,u,udot):
    h_cur,k_cur=params[-1,2:]
    params[-1,:2]=OLS_regression(h_cur,k_cur,u,udot)
    a_cur,b_cur=params[-1,:2]
    SSE_prev=SSE(a_cur,b_cur,h_cur,k_cur,udot)

    for _ in range(iter_limit):
        h_next,k_next=gradient_descent(a_cur,b_cur,h_cur,k_cur,udot)
        a_next,b_next=OLS_regression(h_next,k_next,u,udot)
        SSE_curr=SSE(a_next,b_next,h_next,k_next,udot)

        if SSE_prev-SSE_curr<0:
            break
        
        params=np.concatenate((params,np.array([[a_next,b_next,h_next,k_next]])),axis=0)
        a_cur,b_cur,h_cur,k_cur=a_next,b_next,h_next,k_next
        SSE_prev=SSE_curr
        # print(params[-1],SSE_curr)
    
    return params


# print(gradient_descent(1,1,1,2,udot))
# gradient desent update h and k when a and b is not from OLS regression
# alternating_optimazation(params,u,udot)
# but with optimized a and b given h and k, h and k are not changed