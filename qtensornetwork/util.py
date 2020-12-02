import jax.numpy as np
import numpy as onp
import qtensornetwork.components as qtnc
import math

def dtoq_miles(data):
    return np.array([[math.cos(data[i]), math.sin(data[i])] for i in range(len(data))])

def dtoq_reyes(data):
    return np.array([[1 / math.sqrt(data[i]**2 + 1.0), data[i] / math.sqrt(data[i]**2 + 1.0)] for i in range(len(data))])

def data_to_qubits(x, type=None, func=None):
    if type == "Miles":
        return np.array([dtoq_miles(x[i]) for i in range(x.shape[0])])
    elif type == "Reyes":
        return np.array([dtoq_reyes(x[i]) for i in range(x.shape[0])])
    else:
        if func == None:
            raise ValueError("type or func must be specified")
        return np.array([func(x[i]) for i in range(x.shape[0])])