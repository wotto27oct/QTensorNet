import jax.numpy as np
import numpy as onp
import math

def X():
    return np.array([[0, 1], [1, 0]])

def Y():
    return np.array([[0, -1j], [1j, 0]])

def Z():
    return np.array([[1, 0], [0, -1]])

def H():
    return np.array([[1, 1], [1, -1]]) / math.sqrt(2)

def S():
    return np.array([[1, 0], [0, 1j]])

def T():
    return np.array([[1, 0], [0, math.cos(math.pi/4) + 1j * math.sin(math.pi/4)]])

def Rx(x):
    cosx = np.cos(x[0]/2)
    sinx = np.sin(x[0]/2)
    return np.array([[cosx, -1j*sinx], [-1j*sinx, cosx]])

def Ry(x):
    cosx = np.cos(x[0]/2)
    sinx = np.sin(x[0]/2)
    return np.array([[cosx, -sinx], [sinx, cosx]])

def Rz(x):
    cosx = np.cos(x[0]/2)
    sinx = np.sin(x[0]/2)
    return np.array([[cosx - 1j*sinx, 0], [0, cosx + 1j*sinx]])

def Rx_qulacs(x):
    cosx = np.cos(x[0]/2)
    sinx = np.sin(x[0]/2)
    return np.array([[cosx, 1j*sinx], [1j*sinx, cosx]])

def Ry_qulacs(x):
    cosx = np.cos(x[0]/2)
    sinx = np.sin(x[0]/2)
    return np.array([[cosx, sinx], [-sinx, cosx]])

def Rz_qulacs(x):
    cosx = np.cos(x[0]/2)
    sinx = np.sin(x[0]/2)
    return np.array([[cosx + 1j*sinx, 0], [0, cosx - 1j*sinx]])

def Rphi(x):
    cosx = np.cos(x[0])
    sinx = np.sin(x[0])
    return np.array([[1, 0], [0, cosx + 1j*sinx]])

def XX(x):
    cosx = np.cos(x[0])
    sinx = np.sin(x[0])
    return np.array([[cosx, 0, 0, -1j*sinx],
                    [0, cosx, -1j*sinx, 0],
                    [0, -1j*sinx, cosx, 0],
                    [-1j*sinx, 0, 0, cosx]])

def YY(x):
    cosx = np.cos(x[0])
    sinx = np.sin(x[0])
    return np.array([[cosx, 0, 0, 1j*sinx],
                    [0, cosx, -1j*sinx, 0],
                    [0, -1j*sinx, cosx, 0],
                    [1j*sinx, 0, 0, cosx]])

def ZZ(x):
    cosx = np.cos(x[0])
    sinx = np.sin(x[0])
    return np.array([[cosx + 1j*sinx, 0, 0, 0],
                    [0, cosx - 1j*sinx, 0, 0],
                    [0, 0, cosx - 1j*sinx, 0],
                    [0, 0, 0, cosx + 1j*sinx]])

def CNOT():
    return np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0]])

def SWAP():
    return np.array([[1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1]])

def Toffoli():
    return np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 1, 0]])

def Fredkin():
    return np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1]])