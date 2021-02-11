import jax.numpy as np
import numpy as onp
import opt_einsum as oe
import math
from jax import grad, jit
from scipy import optimize
from jax.interpreters import xla
import time
import copy

class AbstructGate():
    def __init__(self, input_qubits, tensor, name=None):
        if isinstance(input_qubits, int):
            self._input_qubits = [input_qubits]
        else:
            self._input_qubits = input_qubits
        if tensor is not None and tensor.shape[0] != tensor.shape[1]:
            raise ValueError("The tensor must be square matrix.")
        self._tensor = tensor
        self._name = name
        self._input_qubits_num = None
        self._shape = None
        if self._input_qubits is not None:
            self._input_qubits_num = len(self._input_qubits)
            self._shape = [2 for i in range(self._input_qubits_num * 2)]
            if self._tensor is not None and 2**self._input_qubits_num != tensor.shape[0]:
                raise ValueError("The shape of the tensor and the input qubits do not match.")
        elif self._tensor is not None:
            self._shape = [2 for i in range(int(onp.log2(self._tensor.shape[0])) * 2)]

    @property
    def input_qubits(self):
        return self._input_qubits
    
    @input_qubits.setter
    def input_qubits(self, input_qubits):
        if isinstance(input_qubits, int):
            self._input_qubits = [input_qubits]
        else:
            self._input_qubits = input_qubits
        if self._input_qubits is not None:
            self._input_qubits_num = len(self._input_qubits)
            self._shape = [2 for i in range(self._input_qubits_num * 2)]
    
    @property
    def tensor(self):
        return self._tensor
    
    @tensor.setter
    def tensor(self, tensor):
        self._tensor = tensor

    @property
    def shape(self):
        return self._shape

    @property
    def input_qubits_num(self):
        return self._input_qubits_num


class Gate(AbstructGate):
    def __init__(self, input_qubits, params=None, func=None, tensor=None, is_updated=False, train_idx=None, name=None):
        super().__init__(input_qubits, tensor, name)
        if params is not None:
            self._params = np.array(params, ndmin=1)
        else:
            self._params = None
        self._func = func
        
        if self._params is not None and self._func is not None:
            if 2**self._input_qubits_num != self._func(self._params).shape[0]:
                raise ValueError("The shape of the function and the input qubits do not match.")
        if is_updated == True:
            if self._params is None and self._tensor is not None:
                self._params = self._tensor.flatten()
                dim = tensor.shape[0]
                def identity(x):
                    return x.reshape(dim, dim)
                self._func = identity
            
        self._is_updated = is_updated
        self._train_idx = train_idx
        if self._is_updated and self._train_idx is not None:
            raise ValueError("The gate for quantum kernel cannot be updated.")
    
    @property
    def params(self):
        return self._params
    
    @params.setter
    def params(self, params):
        self._params = np.array(params, ndmin=1)
        self._tensor = self._func(self._params)

    @property
    def func(self):
        return self._func
        
    @func.setter
    def func(self, func):
        self._func = func

    @property
    def is_updated(self):
        return self._is_updated
    
    @is_updated.setter
    def is_updated(self, is_updated):
        self._is_updated = is_updated

    @property
    def train_idx(self):
        return self._train_idx
    
    @train_idx.setter
    def train_idx(self, train_idx):
        self._train_idx = train_idx
    
    @property
    def tensor(self):
        if self._tensor is None:
            self._tensor = self._func(self._params)
        return self._tensor

    @tensor.setter
    def tensor(self, tensor):
        if tensor.shape[0] != tensor.shape[1]:
            raise ValueError("The tensor must be square matrix.")
        self._tensor = np.array(tensor, ndmin=1)
        self._params = tensor.flatten()
        dim = tensor.shape[0]
        def identity(x):
            return x.reshape(dim, dim)
        self._func = identity
    
    def get_tensor_from_params(self, params):
        return self._func(params)
    
    def get_tensor_from_train(self, train):
        return self._func([train[self._train_idx]])
    
    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, name):
        self._name = name
    
    def is_unitary(self):
        if self._tensor is None:
            self._tensor = self._func(self._params)

        if np.linalg.norm(np.dot(self._tensor.conj().T, self._tensor) - np.eye(2**self._input_qubits_num)) < 1e-08:
            return True
        return False


class Measurement(AbstructGate):
    def __init__(self, input_qubits=None, tensor=None, name=None):
        super().__init__(input_qubits, tensor, name)
            
    def is_hermitian(self):
        if np.linalg.norm(self._tensor - self._tensor.conj().T) < 1e-08:
            return True
        return False


class State():
    def __init__(self, input_qubits, tensor=None, train_idx=None, name=None):
        self._input_qubits = input_qubits
        self._tensor = tensor
        self._name = name
        self._train_idx = train_idx

    @property
    def input_qubits(self):
        return self._input_qubits
    
    @input_qubits.setter
    def input_qubits(self, input_qubits):
        self._input_qubits = input_qubits

    @property
    def tensor(self):
        return self._tensor
    
    @tensor.setter
    def tensor(self, tensor):
        self._tensor = tensor

    @property
    def train_idx(self):
        return self._train_idx
    
    @train_idx.setter
    def train_idx(self, train_idx):
        self._train_idx = train_idx
    
    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, name):
        self._name = name