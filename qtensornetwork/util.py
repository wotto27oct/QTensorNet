import jax.numpy as np
import numpy as onp
import math

def dtoq_miles(data):
    return np.stack([np.cos(data*np.pi/2.0), np.sin(data*np.pi/2.0)], axis=2)


def dtoq_reyes(data):
    qubit_tensor = np.empty([0, 2])
    for i in range(len(data)):
        x1 = 1 / math.sqrt(data[i]**2 + 1.0)
        x2 = data[i] / math.sqrt(data[i]**2 + 1.0)
        qubit_tensor = np.vstack((qubit_tensor, np.array([x1, x2])))
    return qubit_tensor


def data_to_qubits(x, type=None, func=None):
    if type == "Miles":
        return dtoq_miles(x)
    elif type == "Reyes":
        return dtoq_reyes(x)
    else:
        if func == None:
            raise ValueError("type or func must be specified")
        return np.array([func(x[i]) for i in range(x.shape[0])])