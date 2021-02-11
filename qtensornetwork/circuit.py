import jax.numpy as np
import numpy as onp
import opt_einsum as oe
import math
from jax import grad, jit, vmap, value_and_grad
from scipy import optimize
from jax.interpreters import xla
import time
import copy
import qtensornetwork.ansatz as qtnansatz
import qtensornetwork.optimizer as qtnoptimizer
import qtensornetwork.components as qtnc


class Circuit():
    def __init__(self, qubits_num, name=None):
        self._qubits_num = qubits_num
        self._name = name
        self._init_state = []
        for i in range(self._qubits_num):
            self._init_state.append(qtnc.State([i], np.array([1,0])))
        self._full_init_state = None
        self._gates = []
        self._measurements = []
        self._m_expr = []
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
            self._init_state.append(qtnc.State([i], np.array([1,0])))
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
            cont_tensors = []
            cont_str = ""
            for idx, st in enumerate(self._init_state):
                if idx != 0:
                    cont_str += ","
                if st.tensor == None:
                    print("Warning: some of init states is not defined yet.")
                    return None
                cont_tensors.append(st.tensor.reshape([2 for i in range(len(st.input_qubits))]))
                index_str = ""
                for q in st.input_qubits:
                    index_str += oe.get_symbol(q)
                cont_str += index_str

            self._full_init_state = oe.contract(cont_str, *cont_tensors).flatten()

        return self._full_init_state

    def set_init_state(self, state):
        self._init_state = list(filter(lambda ist: False if not set(ist.input_qubits).isdisjoint(set(state.input_qubits)) else True, self._init_state))
        self._init_state.append(state)
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
        for i in range(len(self._m_expr)):
            self._m_expr[i] = None
        for g in gate:
            if not isinstance(g, qtnc.Gate):
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

    def add_measurement(self, measurement):
        if type(measurement) is not type([]):
            measurement = [measurement]
        for m in measurement:
            if not isinstance(m, qtnc.Measurement):
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
        if measurement_index is None:
            measurement_index = [i for i in range(len(self._measurements))]
        elif type(measurement_index) is not type([]):
            measurement_index = [measurement_index]
        
        for midx in measurement_index:
            self.__set_gate_params()
            if midx >= len(self._measurements):
                raise ValueError("the number of measurement is not enough for this index")
            expects.append(self.get_expectation_value_with_params(None, None, self._gate_params, midx))

        return expects


    def get_expectation_value_with_params(self, qxtrain, xtrain, params, midx):
        # TODO: speedup
        if self._m_expr[midx] == None:
            self._m_expr[midx] = self.__create_m_expr(midx)

        cont_tensors = []

        cont_tensors.extend(self.__get_forward_tensor(qxtrain, xtrain, params))
        # add measurement_tensor
        cont_tensors.append(self._measurements[midx].tensor.reshape(self._measurements[midx].shape))
        cont_tensors.extend(self.__get_adjoint_tensor(qxtrain, xtrain, params))

        ans = self._m_expr[midx](*cont_tensors).real
        return ans

    def __get_forward_tensor(self, qxtrain, xtrain, params):
        cont_tensors = []
        # add qubit_tensor
        for st in self._init_state:
            if st.train_idx is not None:
                cont_tensors.append(qxtrain[st.train_idx])
            else:
                cont_tensors.append(st.tensor)

        # add gate_tensor
        for gate_index, g in enumerate(self._gates):
            if g.is_updated is True:
                cont_tensors.append(g.get_tensor_from_params(params[gate_index]).T.reshape(g.shape))
            elif g.train_idx is not None:
                cont_tensors.append(g.get_tensor_from_params(np.array([xtrain[tidx] for tidx in g.train_idx])).T.reshape(g.shape))
            else:
                cont_tensors.append(g.tensor.T.reshape(g.shape))
        return cont_tensors

    def __get_adjoint_tensor(self, qxtrain, xtrain, params):
        cont_tensors = []
        # add adjoint gate_tensor
        for gate_index, g in reversed(list(enumerate(self._gates))):
            if g.is_updated is True:
                cont_tensors.append(g.get_tensor_from_params(params[gate_index]).conj().reshape(g.shape))
            elif g.train_idx is not None:
                cont_tensors.append(g.get_tensor_from_params(np.array([xtrain[tidx] for tidx in g.train_idx])).conj().reshape(g.shape))
            else:
                cont_tensors.append(g.tensor.conj().reshape(g.shape))
        # add adjoint qubit_tensor
        for st in self._init_state:
            if st.train_idx is not None:
                cont_tensors.append(qxtrain[st.train_idx].conj())
            else:
                cont_tensors.append(st.tensor.conj())

        return cont_tensors
        
    #def __create_m_expr(self, q_index):
    def __create_m_expr(self, midx):
        # TODO: use causal cone
        # l: the maximum number of index one qubit may use
        l = 2 * max(self._layer_nums) + 2
        cont_shapes = []
        cont_indexes = []
        qubit_indexes = [l * i for i in range(self._qubits_num)]
        
        # add qubit_tensor
        for st in self._init_state:
            cont_shapes.append([2**len(st.input_qubits)])
            index_str = ""
            for q in st.input_qubits:
                index_str += oe.get_symbol(qubit_indexes[q])
            cont_indexes.append(index_str)
        
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
        for st in self._init_state:
            cont_shapes.append([2**len(st.input_qubits)])
            index_str = ""
            for q in st.input_qubits:
                index_str += oe.get_symbol(qubit_indexes[q])
            cont_indexes.append(index_str)
        
        cont_str = ""
        for i in range(len(cont_indexes)):
            if i != 0:
                cont_str += ","
            cont_str += cont_indexes[i]
        
        return oe.contract_expression(cont_str, *cont_shapes)

    def get_state_vector(self):
        # l: the maximum number of index one qubit may use
        l = len(self._gates) * 2 + 2
        cont_shapes = []
        cont_indexes = []
        qubit_indexes = [l * i for i in range(self._qubits_num)]
        
        # add qubit_tensor
        for st in self._init_state:
            cont_shapes.append([2**len(st.input_qubits)])
            index_str = ""
            for q in st.input_qubits:
                index_str += oe.get_symbol(qubit_indexes[q])
            cont_indexes.append(index_str)
        
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

        self.__set_gate_params()
        cont_tensors = self.__get_forward_tensor(None, None, self._gate_params)

        return oe.contract(cont_str, *cont_tensors).reshape(-1)

    def __loss(self, params, qxtrain, xtrain, ytrain):
        loss_val = 0.0
        for i in range(len(self._measurements)):
            loss_val += np.linalg.norm(ytrain[i] - self.get_expectation_value_with_params(qxtrain, xtrain, params, i)) ** 2
        
        return loss_val

    def __get_accs(self, params, qxtrain, xtrain, ytrain):
        # for multi (qxtrain, ytrain)
        accs = []
        for i in range(len(self._measurements)):
            accs.append(self.get_expectation_value_with_params(qxtrain, xtrain, params, i))

        return accs

    def fit(self, qxtrain, xtrain, ytrain, qxtest, xtest, ytest, optimizer, epoch=1, batch_size=1, num_data=0, record_tensors=False, show_grad=False):
        self.__set_gate_params()
        params = self._gate_params

        if num_data == 0:
            if qxtrain is not None:
                num_data = qxtrain.shape[0]
            else:
                num_data = xtrain.shape[0]

        if num_data % batch_size != 0:
            print("Recommand: numdata should be divided by batchsize.")

        if ytrain.ndim == 1:
            ytrain = np.array([[ytrain[i]] for i in range(ytrain.shape[0])])
        if ytest.ndim == 1:
            ytest = np.array([[ytest[i]] for i in range(ytest.shape[0])])
        
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
        start = time.time()

        def loss(par, qxtr, xtr, ytr):
            return self.__loss(par, qxtr, xtr, ytr)
        
        loss_value_and_grad = value_and_grad(loss)

        def loss_value_and_grad_args(loss_args):
            par, (qxtr, (xtr, ytr)) = loss_args
            return loss_value_and_grad(par, qxtr, xtr, ytr)

        batched_loss_value_and_grad_args = vmap(loss_value_and_grad_args, ((None, (0, (0, 0)),),))

        @jit
        def batched_loss_value_and_grad(par, qxtr, xtr, ytr):
            return batched_loss_value_and_grad_args((par, (qxtr, (xtr, ytr))))

        whole_start = time.time()
        
        print("compling loss and grad function......")
        start = time.time()
        initial_loss = 0.0
        for idx in range(0, num_data, batch_size):
            batch_x = None if xtrain is None else xtrain[idx: idx+ batch_size if idx + batch_size < num_data else num_data]
            batch_qx = None if qxtrain is None else qxtrain[idx: idx+ batch_size if idx + batch_size < num_data else num_data]
            batch_y = ytrain[idx: idx+ batch_size if idx + batch_size < num_data else num_data]
            batch_loss, _ = batched_loss_value_and_grad(params, batch_qx, batch_x, batch_y)
            initial_loss += sum(batch_loss)
        end = time.time()
        print("time:", end - start)

        print("initial loss:", initial_loss)

        print("------optimization start------")
        start = time.time()

        epoch_tensor = []

        for ep in range(epoch):
            sff_idx = onp.random.permutation(num_data)
            loss = 0
            epoch_grad_list = []
            start2 = time.time()
            for idx in range(0, num_data, batch_size):
                batch_x = None if xtrain is None else xtrain[sff_idx[idx: idx+ batch_size if idx + batch_size < num_data else num_data]]
                batch_qx = None if qxtrain is None else qxtrain[sff_idx[idx: idx+ batch_size if idx + batch_size < num_data else num_data]]
                batch_y = ytrain[sff_idx[idx: idx+ batch_size if idx + batch_size < num_data else num_data]]
                batch_loss, batch_grad = batched_loss_value_and_grad(params, batch_qx, batch_x, batch_y)
                loss += sum(batch_loss)
                grad_params =[]
                for b_grad in batch_grad:
                    if b_grad is None:
                        grad_params.append(None)
                    else:
                        grad_params.append(np.sum(b_grad, axis=0))
                if optimizer.name == "mansgd":
                    def f(par):
                        batch_loss, _ = batched_loss_value_and_grad(par, batch_qx, batch_x, batch_y)
                        return batch_loss
                    def grad_f(par):
                        _, batch_grad = batched_loss_value_and_grad(par, batch_qx, batch_x, batch_y)
                        return batch_grad
                    optimizer.update(params, f, grad_f)
                else:
                    optimizer.update(params, grad_params)
            end2 = time.time()
            print("epoch:", ep + 1, "loss:", loss, 
                "elapsed time for epoch:", end2 - start2)
            epoch_tensor_list = []
            for idx, g in enumerate(self.gates):
                if self._gate_params[idx] is not None:
                    epoch_tensor_list.append(g.func(params[idx]))
            epoch_tensor.append(epoch_tensor_list)
            if show_grad:
                total_idx = 0
                for idx, g in enumerate(self.gates):
                    if self._gate_params[idx] is not None:
                        epoch_gate_grad = onp.asarray(epoch_grad_list)[:,idx]
                        print(f"grad for gate {idx}: {onp.average(onp.average(onp.abs(epoch_gate_grad)))}")

        print("------optimization end------")

        end = time.time()
        print("optimization time:", end - start, "[sec]")
        whole_end = time.time()
        print("whole elapsed time:", whole_end - whole_start, "[sec]")

        #self._gate_params = params
        for idx, g in enumerate(self.gates):
            if self._gate_params[idx] is not None:
                g.params = params[idx]
                self._gate_params[idx] = g.params

        if record_tensors:
            return epoch_tensor

    def classify(self, qxtrain, xtrain, ytrain, qxtest, xtest, ytest, optimizer, epoch=1, batch_size=1, num_data=0, record_tensors=False, show_grad=False):
        self.__set_gate_params()
        params = self._gate_params

        if num_data == 0:
            if qxtrain is not None:
                num_data = qxtrain.shape[0]
            else:
                num_data = xtrain.shape[0]
        
        if num_data % batch_size != 0:
            print("Recommand: numdata should be divided by numdata.")

        if ytrain.ndim == 1:
            ytrain = np.array([[ytrain[i]] for i in range(ytrain.shape[0])])
        if ytest.ndim == 1:
            ytest = np.array([[ytest[i]] for i in range(ytest.shape[0])])
        
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
        start = time.time()

        def loss(par, qxtr, xtr, ytr):
            return self.__loss(par, qxtr, xtr, ytr)
        
        loss_value_and_grad = value_and_grad(loss)

        def loss_value_and_grad_args(loss_args):
            par, (qxtr, (xtr, ytr)) = loss_args
            return loss_value_and_grad(par, qxtr, xtr, ytr)

        batched_loss_value_and_grad_args = vmap(loss_value_and_grad_args, ((None, (0, (0, 0)),),))

        @jit
        def batched_loss_value_and_grad(par, qxtr, xtr, ytr):
            return batched_loss_value_and_grad_args((par, (qxtr, (xtr, ytr))))
        
        def get_accs_args(acc_args):
            par, (qxtr, (xtr, ytr)) = acc_args
            return self.__get_accs(par, qxtr, xtr, ytr)
        batched_get_accs_args = vmap(get_accs_args, ((None, (0, (0, 0)),),))

        @jit
        def batched_get_accs(par, qxtr, xtr, ytr):
            return batched_get_accs_args((par, (qxtr, (xtr, ytr))))

        def get_accuracy(par, qxtr, xtr, ytr):
            acc = 0
            accs = batched_get_accs(par, qxtr, xtr, ytr)
            accs = onp.array(accs).T
            for idx, a in enumerate(accs):
                a = np.array(np.append(a, 1.0-sum(a))) if len(a) < len(ytr[idx]) else np.array(a)
                if np.argmax(ytr[idx]) == onp.argmax(a):
                    acc += 1

            return float(acc) / len(accs)

        whole_start = time.time()
        
        print("compling loss and grad function......")
        start = time.time()
        initial_loss = 0.0
        for idx in range(0, num_data, batch_size):
            batch_x = None if xtrain is None else xtrain[idx: idx+ batch_size if idx + batch_size < num_data else num_data]
            batch_qx = None if qxtrain is None else qxtrain[idx: idx+ batch_size if idx + batch_size < num_data else num_data]
            batch_y = ytrain[idx: idx+ batch_size if idx + batch_size < num_data else num_data]
            batch_loss, _ = batched_loss_value_and_grad(params, batch_qx, batch_x, batch_y)
            initial_loss += sum(batch_loss)
        end = time.time()
        print("time:", end - start)
        print("compiling accuracy function.......")
        start = time.time()
        initial_acc = get_accuracy(params, qxtrain, xtrain, ytrain)
        initial_test_acc = get_accuracy(params, qxtest, xtest, ytest)
        end = time.time()
        print("time:", end - start)

        print("initial loss:", initial_loss)
        print("initial train_acc:", initial_acc)
        print("initial test_acc:", initial_test_acc)

        print("------optimization start------")
        start = time.time()

        epoch_tensor = []

        for ep in range(epoch):
            sff_idx = onp.random.permutation(num_data)
            loss = 0
            epoch_grad_list = []
            start2 = time.time()
            for idx in range(0, num_data, batch_size):
                batch_x = None if xtrain is None else xtrain[sff_idx[idx: idx+ batch_size if idx + batch_size < num_data else num_data]]
                batch_qx = None if qxtrain is None else qxtrain[sff_idx[idx: idx+ batch_size if idx + batch_size < num_data else num_data]]
                batch_y = ytrain[sff_idx[idx: idx+ batch_size if idx + batch_size < num_data else num_data]]
                batch_loss, batch_grad = batched_loss_value_and_grad(params, batch_qx, batch_x, batch_y)
                loss += sum(batch_loss)
                grad_params =[]
                for b_grad in batch_grad:
                    if b_grad is None:
                        grad_params.append(None)
                    else:
                        grad_params.append(np.sum(b_grad, axis=0))
                if optimizer.name == "mansgd":
                    def f(par):
                        batch_loss, _ = batched_loss_value_and_grad(par, batch_qx, batch_x, batch_y)
                        return batch_loss
                    def grad_f(par):
                        _, batch_grad = batched_loss_value_and_grad(par, batch_qx, batch_x, batch_y)
                        return batch_grad
                    optimizer.update(params, f, grad_f)
                else:
                    optimizer.update(params, grad_params)
            end2 = time.time()
            print("epoch:", ep + 1, "loss:", loss, 
                "train_accuracy:", get_accuracy(params, qxtrain, xtrain, ytrain), "test_accuracy:", get_accuracy(params, qxtest, xtest, ytest),
                "elapsed time for epoch:", end2 - start2)
            epoch_tensor_list = []
            for idx, g in enumerate(self.gates):
                if self._gate_params[idx] is not None:
                    epoch_tensor_list.append(g.func(params[idx]))
            epoch_tensor.append(epoch_tensor_list)
            if show_grad:
                total_idx = 0
                for idx, g in enumerate(self.gates):
                    if self._gate_params[idx] is not None:
                        epoch_gate_grad = onp.asarray(epoch_grad_list)[:,idx]
                        print(f"grad for gate {idx}: {onp.average(onp.average(onp.abs(epoch_gate_grad)))}")

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