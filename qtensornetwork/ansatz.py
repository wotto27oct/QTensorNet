import jax.numpy as np
import numpy as onp
import qtensornetwork.components as qtnc
import copy
import math

class BaseAnsatz:
    def __init__(self, q_support, is_updated, name=None):
        self._q_support = q_support
        self._is_updated = is_updated
        self._name = name
        self._gates = []
        self._prior_measure_qubits = q_support

    def allocate_gates(self):
        raise NotImplementedError()

    @property
    def q_support(self):
        return self._q_support
    
    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, name):
        self._name = name
    
    @property
    def gates(self):
        if len(self._gates) == 0:
            self.allocate_gates()
        return self._gates

    @property
    def gates_num(self):
        if len(self._gates) == 0:
            self.allocate_gates()
        return len(self._gates)

    @property
    def prior_measure_qubits(self):
        if len(self._gates) == 0:
            self.allocate_gates()
        return self._prior_measure_qubits
    
    def simple_2(self, x):
        cosx0 = np.cos(x[0]/2)
        sinx0 = np.sin(x[0]/2)
        cosx1 = np.cos(x[1]/2)
        sinx1 = np.sin(x[1]/2)
        return np.array([[cosx0*cosx1, -cosx0*sinx1, -sinx0*cosx1, sinx0*sinx1],
                            [cosx0*sinx1, cosx0*cosx1, -sinx0*sinx1, -sinx0*cosx1],
                            [sinx0*sinx1, sinx0*cosx1, cosx0*sinx1, cosx0*cosx1],
                            [sinx0*cosx1, -sinx0*sinx1, cosx0*cosx1, -cosx0*sinx1]])

class TTN(BaseAnsatz):
    def __init__(self, q_support, is_updated=True, gate_type=None, gate_input_num=None, gate_output_num=None, 
            gate_func=None, gate_params_num=None, name=None):
        super().__init__(q_support, is_updated, name)
        self._gate_input_num = gate_input_num
        self._gate_output_num = gate_output_num
        if self._gate_input_num is not None and self._gate_output_num is not None and self._gate_input_num <= self._gate_output_num:
            raise ValueError("input qubits of gate is smaller than output qubits")
        self._gate_type = gate_type
        self._gate_func = gate_func
        self._gate_params_num = gate_params_num

        if self._gate_type is None:
            if self._gate_func is None and self._gate_params_num is None:
                if self._gate_input_num == 2 and self._gate_output_num == 1:
                    self._gate_type = "simple2to1"
                else:
                    if gate_input_num is None:
                        raise ValueError("gate_input_num is needed.")
                    if gate_output_num is None:
                        raise ValueError("gate_output_num is needed.")
                
        if self._gate_type is not None:
            if self._gate_type == "simple2to1":
                self._gate_input_num = 2
                self._gate_output_num = 1
                self._gate_func = self.simple_2
                self._gate_params_num = 2
            else:
                raise ValueError("This type of gate is not supported.")

    def allocate_gates(self):
        q_input = copy.deepcopy(self._q_support)
        del_qnum = self._gate_input_num - self._gate_output_num
        prior_measure_qubits = []
        def get_gate_params():
            if self._gate_params_num is None:
                return None
            else:
                return np.array(onp.random.randn(self._gate_params_num))
        while True:
            i = 0
            while self._gate_input_num <= len(q_input) - i:
                input_qubits = []
                for j in range(self._gate_input_num):
                    input_qubits.append(q_input[i + j])
                self._gates.append(qtnc.Gate(input_qubits=input_qubits, params=get_gate_params(), 
                    func=self._gate_func, is_updated=self._is_updated))
                prior_measure_qubits.extend(q_input[i:i+del_qnum])
                del q_input[i:i+del_qnum]
                i += self._gate_output_num
            if len(q_input) < self._gate_input_num:
                break
            q_input = q_input[::-1]
        self._prior_measure_qubits =q_input + prior_measure_qubits[::-1]

class TTN2D(BaseAnsatz):
    def __init__(self, q_support, q_width, q_height, is_updated=True, gate_type=None, gate_w_input_num=None, gate_h_input_num=None, gate_w_output_num=None, 
            gate_h_output_num=None, gate_func=None, gate_params_num=None, name=None):
        super().__init__(q_support, is_updated, name)
        self._q_width = q_width
        self._q_height = q_height
        self._gate_w_input_num = gate_w_input_num
        self._gate_h_input_num = gate_h_input_num
        self._gate_w_output_num = gate_w_output_num
        self._gate_h_output_num = gate_h_output_num
        self._gate_input_num = None
        if self._gate_w_input_num is not None and self._gate_h_input_num is not None:
            self._gate_input_num = self._gate_w_input_num * self._gate_h_input_num
        self._gate_output_num = None
        if self._gate_w_output_num is not None and self._gate_h_output_num is not None:
            self._gate_output_num = self._gate_w_output_num * self._gate_h_output_num
        if self._gate_input_num is not None and self._gate_output_num is not None and self._gate_input_num <= self._gate_output_num:
            raise ValueError("input qubits of gate is smaller than output qubits")
        self._gate_type = gate_type
        self._gate_func = gate_func
        self._gate_params_num = gate_params_num

        if self._gate_type is None:
            if self._gate_func is None and self._gate_params_num is None:
                if self._gate_input_num is None:
                    raise ValueError("gate_input_num is needed.")
                if self._gate_output_num is None:
                    raise ValueError("gate_output_num is needed.")

        if self._gate_output_num != 1:
            raise ValueError("gate_output_num must be 1")

        if int(math.log(self._q_width, self._gate_w_input_num)) != int(math.log(self._q_height, self._gate_h_input_num)):
            raise ValueError("the two pair (width, gate_w_input_num) and (height, gate_h_input_num) is not suitable")
                
        if self._gate_type is not None:
            raise ValueError("This type of gate is not supported.")

    def allocate_gates(self):
        q_input = copy.deepcopy(self._q_support)
        q_len = int(math.sqrt(len(q_input)))
        prior_measure_qubits = []
        def get_gate_params():
            if self._gate_params_num is None:
                return None
            else:
                return np.array(onp.random.randn(self._gate_params_num))
        for loop_i in range(int(math.log(self._q_width, self._gate_w_input_num))):
            print(loop_i)
            for h in range(0, q_len, self._gate_h_input_num**(loop_i+1)):
                for w in range(0, q_len, self._gate_w_input_num**(loop_i+1)):
                    input_qubits = []
                    for hin in range(h, h+self._gate_h_input_num**(loop_i+1), self._gate_h_input_num**loop_i):
                        for win in range(w, w+self._gate_w_input_num**(loop_i+1), self._gate_w_input_num**loop_i):
                            input_qubits.append(q_input[hin*q_len + win])
                            if hin != 0 and win != 0:
                                prior_measure_qubits.append(q_input[hin*q_len + win])
                    self._gates.append(qtnc.Gate(input_qubits=input_qubits, params=get_gate_params(), 
                        func=self._gate_func, is_updated=self._is_updated))
        prior_measure_qubits.append(q_input[0])
        self._prior_measure_qubits = prior_measure_qubits[::-1]

class MPS(BaseAnsatz):
    def __init__(self, q_support, is_updated=True, gate_type=None, gate_input_num=None, gate_output_num=None, 
            gate_func=None, gate_params_num=None, name=None):
        super().__init__(q_support, is_updated, name)
        self._gate_type = gate_type
        self._gate_input_num = gate_input_num
        self._gate_output_num = gate_output_num
        if self._gate_input_num is not None and self._gate_output_num is not None and self._gate_input_num <= self._gate_output_num:
            raise ValueError("input qubits of gate is smaller than output qubits")
        self._gate_func = gate_func
        self._gate_params_num = gate_params_num

        if self._gate_type is None:
            if self._gate_func is None and self._gate_params_num is None:
                if self._gate_input_num == 2 and self._gate_output_num == 1:
                    self._gate_type = "simple2to1"
                else:
                    if gate_input_num is None:
                        raise ValueError("gate_input_num is needed.")
                    if gate_output_num is None:
                        raise ValueError("gate_output_num is needed.")
                
        if self._gate_type is not None:
            if self._gate_type == "simple2to1":
                self._gate_input_num = 2
                self._gate_output_num = 1
                self._gate_func = self.simple_2
                self._gate_params_num = 2
            else:
                raise ValueError("This type of gate is not supported.")

    def allocate_gates(self):
        q_input = copy.deepcopy(self._q_support)
        del_qnum = self._gate_input_num - self._gate_output_num
        used_qubits = 0
        def get_gate_params():
            if self._gate_params_num is None:
                return None
            else:
                return np.array(onp.random.randn(self._gate_params_num))
        for i in range(0, len(q_input) - self._gate_input_num + 1, del_qnum):
            input_qubits = [q_input[i + j] for j in range(self._gate_input_num)]
            used_qubits = input_qubits[-1]
            self._gates.append(qtnc.Gate(input_qubits=input_qubits, params=get_gate_params(), 
                func=self._gate_func, is_updated=self._is_updated))

        q_input = q_input[:used_qubits+1]
        self._prior_measure_qubits = q_input[::-1]

class MERA(BaseAnsatz):
    def __init__(self, q_support, is_updated=True, gate_type=None, gate_input_num=None, gate_output_num=None,
                gate_func=None, gate_params_num=None, disentangler_type=None, disentangler_func=None,
                disentangler_params_num=None, name=None):
        super().__init__(q_support, is_updated, name)
        self._gate_input_num = gate_input_num
        self._gate_output_num = gate_output_num
        if self._gate_input_num is not None and self._gate_output_num is not None and self._gate_input_num <= self._gate_output_num:
            raise ValueError("input qubits of gate is smaller than output qubits")
        self._gate_type = gate_type
        self._gate_func = gate_func
        self._gate_params_num = gate_params_num
        self._disentangler_type = disentangler_type
        self._disentangler_func = disentangler_func
        self._disentangler_params_num = disentangler_params_num

        if self._gate_type is None:
            if self._gate_func is None and self._gate_params_num is None:
                if self._gate_input_num == 2 and self._gate_output_num == 1:
                    self._gate_type = "simple2to1"
                else:
                    if gate_input_num is None:
                        raise ValueError("gate_input_num is needed.")
                    if gate_output_num is None:
                        raise ValueError("gate_output_num is needed.")
        if self._gate_type is not None:
            if self._gate_type == "simple2to1":
                self._gate_input_num = 2
                self._gate_output_num = 1
                self._gate_func = self.simple_2
                self._gate_params_num = 2
            else:
                raise ValueError("This type of gate is not supported.")

        if self._disentangler_type is None:
            if self._disentangler_func is None and self._disentangler_params_num is None:
                self._disentangler_type = "simple"
        if self._disentangler_type is not None:
            if disentangler_type == "simple":
                self._disentangler_func = self.simple_2
                self._disentangler_params_num = 2
            else:
                raise ValueError("This type of disentangler is not supported.")

    def allocate_gates(self):
        q_input = copy.deepcopy(self._q_support)
        del_qnum = self._gate_input_num - self._gate_output_num
        prior_measure_qubits = []
        gate_index = 0
        disentangler_index = 0
        def get_disentangler_params():
            if self._disentangler_params_num is None:
                return None
            else:
                return np.array(onp.random.randn(self._disentangler_params_num))
        def get_gate_params():
            if self._gate_params_num is None:
                return None
            else:
                return np.array(onp.random.randn(self._gate_params_num))
        while True:
            for i in range(self._gate_input_num - 1, len(q_input) - 1, self._gate_input_num):
                input_qubits = [q_input[i], q_input[i+1]]
                self._gates.append(qtnc.Gate(input_qubits=input_qubits, params=get_disentangler_params(),
                    func =self._disentangler_func, is_updated=self._is_updated, name="disentangler"))
                disentangler_index += 1
            i = 0
            while self._gate_input_num <= len(q_input) - i:
                input_qubits = []
                for j in range(self._gate_input_num):
                    input_qubits.append(q_input[i + j])
                self._gates.append(qtnc.Gate(input_qubits=input_qubits, params=get_gate_params(),
                    func=self._gate_func, is_updated=self._is_updated, name="gate"))
                gate_index += 1
                prior_measure_qubits.extend(q_input[i:i+del_qnum])
                del q_input[i:i+del_qnum]
                i += self._gate_output_num
            if len(q_input) < self._gate_input_num:
                break
            q_input = q_input[::-1]

        self._prior_measure_qubits =q_input + prior_measure_qubits[::-1]

    @property
    def TTNgates(self):
        if len(self._gates) == 0:
            self.allocate_gates()
        return [g for g in self._gates if g.name=="gate"]
    
    @property
    def disentanglers(self):
        if len(self._gates) == 0:
            self.allocate_gates()
        return [g for g in self._gates if g.name=="disentangler"]