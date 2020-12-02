import jax.numpy as np
import numpy as onp
import math
import qtensornetwork.components as qtc
import qtensornetwork.gate_util as qtg_util
import opt_einsum as oe
from jax import jit


class H(qtc.Gate):
    def __init__(self, input_qubits, name=None):
        super().__init__(input_qubits, tensor=qtg_util.H(), name=name)

class X(qtc.Gate):
    def __init__(self, input_qubits, name=None):
        super().__init__(input_qubits, tensor=qtg_util.X(), name=name)

class Y(qtc.Gate):
    def __init__(self, input_qubits, name=None):
        super().__init__(input_qubits, tensor=qtg_util.Y(), name=name)

class Z(qtc.Gate):
    def __init__(self, input_qubits, name=None):
        super().__init__(input_qubits, tensor=qtg_util.Z(), name=name)

class RX(qtc.Gate):
    def __init__(self, input_qubits, params, is_updated=False, name=None):
        super().__init__(input_qubits, params, func=qtg_util.Rx, is_updated=is_updated, name=name)

class RY(qtc.Gate):
    def __init__(self, input_qubits, params, is_updated=False, name=None):
        super().__init__(input_qubits, params, func=qtg_util.Ry, is_updated=is_updated, name=name)

class RZ(qtc.Gate):
    def __init__(self, input_qubits, params, is_updated=False, name=None):
        super().__init__(input_qubits, params, func=qtg_util.Rz, is_updated=is_updated, name=name)

class RX_qulacs(qtc.Gate):
    def __init__(self, input_qubits, params, is_updated=False, name=None):
        super().__init__(input_qubits, params, func=qtg_util.Rx_qulacs, is_updated=is_updated, name=name)

class RY_qulacs(qtc.Gate):
    def __init__(self, input_qubits, params, is_updated=False, name=None):
        super().__init__(input_qubits, params, func=qtg_util.Ry_qulacs, is_updated=is_updated, name=name)

class RZ_qulacs(qtc.Gate):
    def __init__(self, input_qubits, params, is_updated=False, name=None):
        super().__init__(input_qubits, params, func=qtg_util.Rz_qulacs, is_updated=is_updated, name=name)

class Phase(qtc.Gate):
    def __init__(self, input_qubits, params, is_updated=False, name=None):
        super().__init__(input_qubits, params, func=qtg_util.Rphi, is_updated=is_updated, name=name)

class S(qtc.Gate):
    def __init__(self, input_qubits, name=None):
        super().__init__(input_qubits, tensor=qtg_util.S(), name=name)

class T(qtc.Gate):
    def __init__(self, input_qubits, name=None):
        super().__init__(input_qubits, tensor=qtg_util.T(), name=name)

class CNOT(qtc.Gate):
    def __init__(self, input_qubits, name=None):
        super().__init__(input_qubits, tensor=qtg_util.CNOT(), name=name)

class SWAP(qtc.Gate):
    def __init__(self, input_qubits, name=None):
        super().__init__(input_qubits, tensor=qtg_util.SWAP(), name=name)

class Toffoli(qtc.Gate):
    def __init__(self, input_qubits, name=None):
        super().__init__(input_qubits, tensor=qtg_util.Toffoli(), name=name)

class Fredkin(qtc.Gate):
    def __init__(self, input_qubits, name=None):
        super().__init__(input_qubits, tensor=qtg_util.Fredkin(), name=name)

class XX(qtc.Gate):
    def __init__(self, input_qubits, params, is_updated=False, name=None):
        super().__init__(input_qubits, params, func=qtg_util.XX, is_updated=is_updated, name=name)

class YY(qtc.Gate):
    def __init__(self, input_qubits, params, is_updated=False, name=None):
        super().__init__(input_qubits, params, func=qtg_util.YY, is_updated=is_updated, name=name)

class ZZ(qtc.Gate):
    def __init__(self, input_qubits, params, is_updated=False, name=None):
        super().__init__(input_qubits, params, func=qtg_util.ZZ, is_updated=is_updated, name=name)

def controlled_gate(gate, c_qubit):
    input_qubits = [c_qubit] + gate.input_qubits
    if gate.func is None:
        U = gate.tensor
        sh, _ = U.shape
        I = np.eye(sh)
        zero0 = np.zeros((sh,sh))
        zero1 = np.zeros((sh,sh))
        A = np.hstack([I, zero0])
        B = np.hstack([zero1, U])
        gate = qtc.Gate(input_qubits, tensor=np.vstack([A, B]))
        return gate
    else:
        def func(params):
            U = gate.func(params)
            sh, _ = U.shape
            I = np.eye(sh)
            zero0 = np.zeros((sh,sh))
            zero1 = np.zeros((sh,sh))
            A = np.hstack([I, zero0])
            B = np.hstack([zero1, U])
            return np.vstack([A, B])

        func_jit = jit(func)
        res = qtc.Gate(input_qubits, params=gate.params, func=func_jit)
        return res


def combine_gates(gates):
    input_qubits = []
    qubit_indexes = []
    cont_indexes = []
    cont_shapes = []
    params = []
    for gate in gates:
        cont_shapes.append(gate.shape)
        index_str = ""
        for q in gate.input_qubits:
            if q not in input_qubits:
                input_qubits.append(q)
                qubit_indexes.append(2 * len(gates) * (len(input_qubits)-1))
            ind = input_qubits.index(q)
            index_str += oe.get_symbol(qubit_indexes[ind])
            qubit_indexes[ind] += 1
        for q in gate.input_qubits:
            ind = input_qubits.index(q)
            index_str += oe.get_symbol(qubit_indexes[ind])
        cont_indexes.append(index_str)
        if gate.params is not None:
            params.extend(gate.params)

    cont_str = ""
    for i in range(len(cont_indexes)):
        if i != 0:
            cont_str += ","
        cont_str += cont_indexes[i]
    
    cont_out_str = ""
    for i in range(len(input_qubits)):
        cont_out_str += oe.get_symbol(2*len(gates)*i)
    for i in range(len(input_qubits)):
        cont_out_str += oe.get_symbol(qubit_indexes[i])
    cont_str = cont_str + "->" + cont_out_str

    if params is None:
        cont_tensors = []
        for gate in gates:
            cont_tensors.append(gate.tensor.T.reshape(gate.shape))
        tensor = oe.contract(cont_str, *cont_tensors).reshape(2**len(input_qubits), -1).T
        gate = qtc.Gate(input_qubits, tensor=tensor)
        return gate

    else:
        def func(params):
            cont_tensors = []
            idx = 0
            for gate in gates:
                if gate.params is not None:
                    cont_tensors.append(gate.get_tensor_from_params(params[idx:idx+len(gate.params)]).T.reshape(gate.shape))
                    idx += len(gate.params)
                else:
                    cont_tensors.append(gate.tensor.T.reshape(gate.shape))
            return oe.contract(cont_str, *cont_tensors).reshape(2**len(input_qubits), -1).T
        
        func_jit = jit(func)
        params = np.array(params)
        gate = qtc.Gate(input_qubits, params=params, func=func_jit)
        return gate