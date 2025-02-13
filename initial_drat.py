import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


import matplotlib
matplotlib.use("TkAgg")


num = random.randint(1,1001)

Ne = 800;

re = np.random.rand(Ne,1)




Ni = 200;

ri = np.random.rand(Ni, 1)


a = np.vstack((0.02 * np.ones((Ne, 1)), 0.02 + 0.08*ri))




b = np.vstack((0.2 * np.ones((Ne, 1)), 0.25 - 0.05*ri))

c = np.vstack((-65 + 15*re**2, -65*np.ones((Ni, 1))))
# c = np.zeros((Ne+Ni,1))

# c = -20 * np.ones((Ne+Ni, 1))

d = np.vstack((8 - 6*re**2, 2*np.ones((Ni, 1))))



S = np.hstack((0.5 * np.random.rand(Ne+Ni, Ne), -np.random.rand(Ni+Ne, Ni)))

v0 = -65 * np.ones((Ne+Ni, 1))

u0 = b*v0

firings = np.empty((0,2))


def system(t, z):
    global firings

    I = np.vstack((5 + 5 * np.random.randn(Ne,1), 2 + 2 * np.random.randn(Ni,1)))

    v, u = np.split(z, 2)  
    v = v.reshape(-1, 1)  
    u = u.reshape(-1, 1)

    dvdt = 0.04 * (v**2) + 5 * v + 140 - u + I
    dudt = a * (b * v - u)

    spiked_neurons = v>=30
    spike_indice = np.where(spiked_neurons)[0]

    v[spike_indice] = c[spike_indice]
    u[spike_indice] += d[spike_indice]



    I += np.sum(S[:, spike_indice],1, keepdims=True)

    if  spike_indice.size> 0:  
        new_firings = np.column_stack([t * np.ones_like(spike_indice), spike_indice])
        firings = np.vstack([firings, new_firings])


    return np.concatenate((dvdt, dudt)).flatten()



z0 = np.concatenate((v0, u0)).flatten()

t_span = (0, 1000)  
t_eval = np.linspace(0, 1000, 10000)  


sol = solve_ivp(system, t_span, z0, t_eval=t_eval)



v_val = sol.y[num, :]


plt.plot(sol.t, v_val)
plt.title(f"Neuron : {num}")
plt.xlabel("Time(in ms)")
plt.ylabel("Voltage")
plt.show()


print(firings[:, 0], firings[:, 1])

plt.scatter(firings[:, 0], firings[:, 1], s=5)
plt.title("Raster Plot")
plt.xlabel("Time(in ms)")
plt.ylabel("Neuron Number")
plt.show()