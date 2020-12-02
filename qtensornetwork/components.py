""" 
    This software uses opt_einsum.
    Copyright (c) 2014 Daniel Smith
    Released under the MIT license
    https://github.com/dgasmith/opt_einsum/blob/master/LICENSE
"""

import jax.numpy as np
import numpy as onp
import opt_einsum as oe
import math
from jax import grad, jit
from jax.interpreters import xla
import time
import copy

import qtensornetwork.ansatz as qtnansatz
import qtensornetwork.optimizer as qtnoptimizer

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
    def __init__(self, input_qubits, params=None, func=None, tensor=None, is_updated=False, name=None):
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


class Circuit():
    def __init__(self, qubits_num, name=None):
        self._qubits_num = qubits_num
        self._name = name
        self._init_state = [np.array([1, 0])] * qubits_num
        self._full_init_state = None
        self._gates = []
        self._measurements = []
        self._m_expr = []
        self._get_forward_tensor = None
        self._get_adjoint_tensor = None
        self._prior_measure_qubits = [i for i in range(qubits_num)]
        self._measurement_order = []
        self._gate_params = []
        self._layer_nums = [0 for i in range(qubits_num)]

    @property
    def qubits_num(self):
        return self._qubits_num
    
    @qubits_num.setter
    def qubits_num(self, qubits_num):
        if qubits_num < self._qubits_num:
            raise ValueError("the number of qubits can't be reduced.")
        for i in range(self._qubits_num, qubits_num):
            self._init_state.append(np.array([1, 0]))
            self._measurements.append(None)
            self._m_expr.append(None)
            self._prior_measure_qubits.append(i)
        self._qubits_num = qubits_num

    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, name):
        self._name = name

    @property
    def init_state(self):
        return self._init_state

    def get_full_init_state(self):
        if self._full_init_state is None:
            self._full_init_state = self._init_state[0]
            for i in range(1, self._qubits_num):
                self._full_init_state = np.kron(self._full_init_state, self._init_state[i])
        return self._full_init_state
    
    def set_init_state(self, tensor, index=0):
        if type(tensor) is not type([]):
            tensor = [tensor]
        if index + len(tensor) > self._qubits_num:
            raise ValueError("the number of qubits is not enough for this index", index)
        for t in tensor:
            if t.shape[0] != 2 or t.ndim != 1:
                raise ValueError("the initial qubit must be two-dimentional vector")
            self._init_state[index] = t
            index += 1
        self._full_init_state = None
    
    @property
    def prior_measure_qubits(self):
        return self._prior_measure_qubits

    def set_prior_measure_qubits(self, prior_measure_qubits):
        if type(prior_measure_qubits) is not type([]):
            prior_measure_qubits = [prior_measure_qubits]
        if len(prior_measure_qubits) > self._qubits_num:
            raise ValueError("the number of qubits is not enough for the prior measure qubits.")
        s = set([i for i in range(self._qubits_num)])
        self._prior_measure_qubits = []
        for prior in prior_measure_qubits:
            if prior > self._qubits_num:
                raise ValueError("the number of qubits is not enough for the prior measure qubits.")
            self._prior_measure_qubits.append(prior)
            s.remove(prior)
        for i in range(len(s)):
            self._prior_measure_qubits.append(s.pop())

    @property
    def gate_params(self):
        self.__set_gate_params()
        return self._gate_params

    def __set_gate_params(self):
        self._gate_params = []
        for g in self._gates:
            if g.is_updated == True:
                self._gate_params.append(g.params)
            else:
                self._gate_params.append(None)

    @property
    def gates(self):
        return self._gates
    
    def add_gate(self, gate):
        if type(gate) is not type([]):
            gate = [gate]
        for g in gate:
            if not isinstance(g, Gate):
                raise ValueError("the input of function add_gate must be the single or the list of Gate class")
            for i in g.input_qubits:
                if i >= self._qubits_num:
                    raise ValueError("the number of qubits is not enough for the Gate", g.input_qubits)
                self._layer_nums[i] += 1
            self._gates.append(g)
            if g.is_updated is True:
                self._gate_params.append(g.params)
            else:
                self._gate_params.append(None)
        self.__delete_tensor_information()

    def add_measurement(self, measurement):
        if type(measurement) is not type([]):
            measurement = [measurement]
        for m in measurement:
            if not isinstance(m, Measurement):
                raise ValueError("the input of function add_measurement must be the single or the list of Measurement class")
            # m.input_qubits should be the list of single value
            if m.input_qubits is None:
                input_q = []
                for i in range(int(onp.log2(m.tensor.shape[0]))):
                    input_q.append(self._prior_measure_qubits[i])
                m.input_qubits = input_q
            for i in m.input_qubits:
                if i >= self._qubits_num:
                    raise ValueError("the number of qubits is not enough for the measurement", m.input_qubits)
                self._layer_nums[i] += 1
        
            self._measurements.append(m)
            self._m_expr.append(None)

    def append_layer(self, ansatz):
        if not isinstance(ansatz, qtnansatz.BaseAnsatz):
            raise ValueError("input must be Ansatz Class")
        self.add_gate(ansatz.gates)
        self._prior_measure_qubits = ansatz.prior_measure_qubits
        self.__delete_tensor_information()

    def show_circuit_structure(self):
        num_len = len(str(len(self._gates)))
        if num_len <= 1:
            num_len = 2
        wire_str = ""
        for i in range(num_len):
            wire_str += "-"
        circuit_str = [[str(i).rjust(num_len) + ":| >"] for i in range(self._qubits_num)]
        for index, g in enumerate(self._gates):
            append_index = 0
            for input_qubit in g.input_qubits:
                append_index = max(append_index, len(circuit_str[input_qubit]))
            for input_qubit in g.input_qubits:
                while len(circuit_str[input_qubit]) < append_index:
                    circuit_str[input_qubit].append(wire_str)
                circuit_str[input_qubit].append(wire_str)
                circuit_str[input_qubit].append(str(index).rjust(num_len))

        max_layer = 0
        for st in circuit_str:
            max_layer = max(max_layer, len(st))
            
        for index, st in enumerate(circuit_str):
            while len(st) < max_layer:
                st.append(wire_str)

        for index, m in enumerate(self._measurements):
            for input_qubit in m.input_qubits:
                circuit_str[input_qubit].append(wire_str)
                circuit_str[input_qubit].append("|" + str(index) + "|")
        
        for index, st in enumerate(circuit_str):
            print("".join(st))

    def get_expectation_value(self, measurement_index=None):
        expects = []
        self.__delete_tensor_information()
        if measurement_index is None:
            measurement_index = [i for i in range(len(self._measurements))]
        elif type(measurement_index) is not type([]):
            measurement_index = [measurement_index]
        
        for midx in measurement_index:
            self.__set_gate_params()
            if midx >= len(self._measurements):
                raise ValueError("the number of measurement is not enough for this index")
            expects.append(self.get_expectation_value_with_params(self._init_state, self._gate_params, midx))

        return expects


    def get_expectation_value_with_params(self, init_states, params, midx):
        if self._m_expr[midx] == None:
            self._m_expr[midx] = jit(self.__create_m_expr(midx))

        if self._get_forward_tensor == None:
            self.__create_get_forward_tensor()
        
        if self._get_adjoint_tensor == None:
            self.__create_get_adjoint_tensor()

        cont_tensors = []

        cont_tensors.extend(self._get_forward_tensor(init_states, params))

        # add measurement_tensor
        cont_tensors.append(self._measurements[midx].tensor.reshape(self._measurements[midx].shape))
        cont_tensors.extend(self._get_adjoint_tensor(init_states, params))

        ans = self._m_expr[midx](*cont_tensors).real
        return ans

    def __create_get_forward_tensor(self):
        def get_forward_tensor(init_states, params):
            cont_tensors = []
            # add qubit_tensor
            for i in range(self._qubits_num):
                cont_tensors.append(init_states[i])
            # add gate_tensor
            for gate_index, g in enumerate(self._gates):
                if g.is_updated is True:
                    cont_tensors.append(g.get_tensor_from_params(params[gate_index]).T.reshape(g.shape))
                else:
                    cont_tensors.append(g.tensor.T.reshape(g.shape))
            return cont_tensors
        self._get_forward_tensor = get_forward_tensor

    def __create_get_adjoint_tensor(self):
        def get_adjoint_tensor(init_states, params):
            cont_tensors = []
            # add adjoint gate_tensor
            for gate_index, g in reversed(list(enumerate(self._gates))):
                if g.is_updated is True:
                    cont_tensors.append(g.get_tensor_from_params(params[gate_index]).conj().reshape(g.shape))
                else:
                    cont_tensors.append(g.tensor.conj().reshape(g.shape))
            # add adjoint qubit_tensor
            for i in range(self._qubits_num):
                cont_tensors.append(init_states[i].conj())
            return cont_tensors
        self._get_adjoint_tensor = get_adjoint_tensor


    def __delete_tensor_information(self):
        self._get_forward_tensor = None
        self._get_adjoint_tensor = None
        
    #def __create_m_expr(self, q_index):
    def __create_m_expr(self, midx):
        l = 2 * max(self._layer_nums) + 2
        cont_shapes = []
        cont_indexes = []
        qubit_indexes = [l * i for i in range(self._qubits_num)]
        
        # add qubit_tensor
        for i in range(self._qubits_num):
            cont_shapes.append([2])
            cont_indexes.append(oe.get_symbol(qubit_indexes[i]))
        
        # add gate_tensor
        for gate in self._gates:
            cont_shapes.append(gate.shape)
            index_str = ""
            for q in gate.input_qubits:
                index_str += oe.get_symbol(qubit_indexes[q])
                qubit_indexes[q] += 1
            for q in gate.input_qubits:
                index_str += oe.get_symbol(qubit_indexes[q])
            cont_indexes.append(index_str)
        
        # add measurement
        cont_shapes.append(self._measurements[midx].shape)
        index_str = ""
        for q in self._measurements[midx].input_qubits:
            index_str += oe.get_symbol(qubit_indexes[q])
            qubit_indexes[q] += 1
        for q in self._measurements[midx].input_qubits:
            index_str += oe.get_symbol(qubit_indexes[q])
        cont_indexes.append(index_str)

        # add adjoint gate_tensor
        for gate in reversed(self._gates):
            cont_shapes.append(gate.shape)
            index_str = ""
            for q in gate.input_qubits:
                index_str += oe.get_symbol(qubit_indexes[q])
                qubit_indexes[q] += 1
            for q in gate.input_qubits:
                index_str += oe.get_symbol(qubit_indexes[q])
            cont_indexes.append(index_str)

        # add adjoint qubit_tensor
        for i in range(self._qubits_num):
            cont_shapes.append([2])
            cont_indexes.append(oe.get_symbol(qubit_indexes[i]))
        
        cont_str = ""
        for i in range(len(cont_indexes)):
            if i != 0:
                cont_str += ","
            cont_str += cont_indexes[i]
        
        return oe.contract_expression(cont_str, *cont_shapes)

    def get_state_vector(self):
        self.__delete_tensor_information()
        # l: the maximum number of index one qubit may use
        l = len(self._gates) * 2 + 2
        cont_shapes = []
        cont_indexes = []
        qubit_indexes = [l * i for i in range(self._qubits_num)]
        
        # add qubit_tensor
        for i in range(self._qubits_num):
            cont_shapes.append([2])
            cont_indexes.append(oe.get_symbol(qubit_indexes[i]))
        
        # add gate_tensor
        for gate in self._gates:
            cont_shapes.append(gate.shape)
            index_str = ""
            for q in gate.input_qubits:
                index_str += oe.get_symbol(qubit_indexes[q])
                qubit_indexes[q] += 1
            for q in gate.input_qubits:
                index_str += oe.get_symbol(qubit_indexes[q])
            cont_indexes.append(index_str)
        
        cont_str = ""
        for i in range(len(cont_indexes)):
            if i != 0:
                cont_str += ","
            cont_str += cont_indexes[i]
        cont_out_str = ""
        for i in range(self._qubits_num):
            cont_out_str += oe.get_symbol(qubit_indexes[i])
        cont_str = cont_str + "->" + cont_out_str

        if self._get_forward_tensor is None:
            self.__create_get_forward_tensor()
        self.__set_gate_params()
        cont_tensors = self._get_forward_tensor(self._init_state, self._gate_params)

        return oe.contract(cont_str, *cont_tensors).reshape(-1)

    def __loss(self, params, qxtrain, ytrain, qx_order):
        loss_val = 0.0
        for index, qxt in enumerate(qxtrain):
            init_states = self._init_state
            for i, q in enumerate(qx_order):
                init_states[q] = qxt[i]
            if ytrain.ndim == 1:
                loss_val += np.linalg.norm(ytrain[index] - self.get_expectation_value_with_params(init_states, params, 0)) ** 2
            else:
                for i, yt in enumerate(ytrain[index]):
                    loss_val += np.linalg.norm(yt- self.get_expectation_value_with_params(init_states, params, i)) ** 2
        
        return loss_val

    def __get_accs(self, params, qxtrain, ytrain, qx_order):
        accs = []
        for index, qxt in enumerate(qxtrain):
            init_states = self._init_state
            for i, q in enumerate(qx_order):
                init_states[q] = qxt[i]
            if ytrain.ndim == 1:
                exp = self.get_expectation_value_with_params(init_states, params, 0)
                if (ytrain[index] == 0 and exp < 0.5) or (ytrain[index] == 1 and exp > 0.5):
                    accs.append(1)
                else:
                    accs.append(0)
            else:
                accs_index = []
                for i, yt in enumerate(ytrain[index]):
                    accs_index.append(self.get_expectation_value_with_params(init_states, params, i))
                if np.argmax(ytrain[index]) == onp.argmax(onp.array(accs_index)):
                    accs.append(1)
                else:
                    accs.append(0)
        
        return accs

    def fit(self, qxtrain, ytrain, qxtest, ytest, optimizer, epoch=1, batch_size=1, als=False, qx_support=None, record_tensors=False):
        self.__set_gate_params()
        params = self._gate_params
        num_data = qxtrain.shape[0]

        if optimizer == "sgd" or optimizer == "SGD":
            optimizer = qtnoptimizer.SGD()
        elif optimizer == "adam" or optimizer == "Adam":
            optimizer = qtnoptimizer.Adam()
        elif optimizer == "adam_ngd":
            optimizer = qtnoptimizer.Adam_NGD()
        elif optimizer == "radam":
            optimizer = qtnoptimizer.RAdam()
        elif optimizer == "mansgd":
            optimizer = qtnoptimizer.ManSGD()

        if qx_support is not None:
            qx_order = qx_support
        else:
            qx_order = [i for i in range(qxtrain[0].shape[0])]

        start = time.time()
        
        def loss(params, qxtrain, ytrain):
            return self.__loss(params, qxtrain, ytrain, qx_order)
        loss_jax = jit(loss)
        grad_loss_jax = jit(grad(loss_jax))

        def get_accs(params, qxtrain, ytrain):
            return self.__get_accs(params, qxtrain, ytrain, qx_order)
        get_accs_jax = get_accs
        def get_accuracy(params, qxt, yt):
            acc = 0
            accs = get_accs_jax(params, qxt, yt)
            for a in accs:
                if a == 1:
                    acc += 1
            return float(acc) / len(accs)

        whole_start = time.time()

        print("compiling loss function......")
        start = time.time()
        initial_loss = 0.0
        for idx, _ in enumerate(qxtrain):
            initial_loss += loss_jax(params, qxtrain[idx:idx+1], ytrain[idx:idx+1])
        end = time.time()
        print("time:", end - start)
        print("compiling grad function......")
        start = time.time()
        grad_loss_jax(params, qxtrain[0:1], ytrain[0:1])
        end = time.time()
        print("time:", end - start)
        print("compiling accuracy function.......")
        start = time.time()
        initial_acc = get_accuracy(params, qxtrain, ytrain)
        end = time.time()
        print("time:", end - start)

        print("initial loss:", initial_loss)
        print("initial train_acc:", initial_acc)

        print("------optimization start------")
        start = time.time()

        epoch_tensor = []

        if als:
            for ep in range(epoch):
                for gate_idx, g in enumerate(self._gates):
                    if g.is_updated == False:
                        continue
                    sff_idx = onp.random.permutation(num_data)
                    loss = 0
                    for idx in range(0, num_data, batch_size):
                        batch_qx = qxtrain[sff_idx[idx: idx+ batch_size if idx + batch_size < num_data else num_data]]
                        batch_y = ytrain[sff_idx[idx: idx+ batch_size if idx + batch_size < num_data else num_data]]
                        loss += loss_jax(params, batch_qx, batch_y)
                        grad_params = grad_loss_jax(params, batch_qx, batch_y)
                        optimizer.update_with_index(params, grad_params, gate_idx)
                    print("epoch:", ep + 1, "gate:", gate_idx, "loss:", loss, 
                        "train_accuracy:", get_accuracy(params, qxtrain, ytrain), "test_accuracy:", get_accuracy(params, qxtest, ytest))
        else:
            for ep in range(epoch):
                sff_idx = onp.random.permutation(num_data)
                loss = 0
                for idx in range(0, num_data, batch_size):
                    batch_qx = qxtrain[sff_idx[idx: idx+ batch_size if idx + batch_size < num_data else num_data]]
                    batch_y = ytrain[sff_idx[idx: idx+ batch_size if idx + batch_size < num_data else num_data]]
                    loss += loss_jax(params, batch_qx, batch_y)
                    grad_params = grad_loss_jax(params, batch_qx, batch_y)
                    if optimizer.name == "mansgd":
                        def f(params):
                            return loss_jax(params, batch_qx, batch_y)
                        def grad_f(params):
                            return grad_loss_jax(params, batch_qx, batch_y)
                        optimizer.update(params, f, grad_f)
                    else:
                        optimizer.update(params, grad_params)
                print("epoch:", ep + 1, "loss:", loss, 
                    "train_accuracy:", get_accuracy(params, qxtrain, ytrain), "test_accuracy:", get_accuracy(params, qxtest, ytest))
                epoch_tensor_list = []
                for idx, g in enumerate(self.gates):
                    if self._gate_params[idx] is not None:
                        epoch_tensor_list.append(g.func(params[idx]))
                epoch_tensor.append(epoch_tensor_list)

        print("------optimization end------")

        end = time.time()
        print("optimization time:", end - start, "[sec]")
        whole_end = time.time()
        print("whole elapsed time:", whole_end - whole_start, "[sec]")

        for idx, g in enumerate(self.gates):
            if self._gate_params[idx] is not None:
                g.params = params[idx]
                self._gate_params[idx] = g.params

        if record_tensors:
            return epoch_tensor