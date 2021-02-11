# QTensorNet : Tensor-network backend Quantum Circuit Simulator

**QTensorNet** is Quantum Circuit & Quantum Machine Learning Simulator using Tensor Network.

We adopt **tensor-network backend**, so for some specific circuits, QTensorNet can calculate the expectation of the obserbable faster than other simulators.

Moreover, QTensorNet is specialized in Quantum Circuit Learning. You can try Quantum machine learning without any knowledge about quantum mechanics and tensor network.

**Note: This product is yet under develepment (alpha version). We don't guantee any calculation.**

# install
Download folder `qtensornetwork`.

# requirements
Python library [`jax, jaxlib`](https://github.com/google/jax), [`opt_einsum`](https://github.com/dgasmith/opt_einsum) is needed.

# Documents
Preparing now.

# Example1
This example classifies binary `mnist dataset`.

```
import jax.numpy as np
import qtensornetwork.components
import qtensornetwork.circuit
import qtensornetwork.ansatz
import qtensornetwork.util
import qtensornetwork.optimizer
from qtensornetwork.gate import *

from jax.config import config
config.update("jax_enable_x64", True)

```

First, import library `jax.numpy`, which has the same function as the original `numpy`.
`QTensorNet` is placed in `qtensornetwork` folder.

```
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def generate_binary_mnist(f_label, s_label, train_num, test_num, width, height):
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    df = pd.DataFrame(columns=["label"])
    df["label"] = y_train.reshape([-1])
    list_f = df.loc[df.label==f_label].sample(n=train_num)
    list_s = df.loc[df.label==s_label].sample(n=train_num)
    label_list = pd.concat([list_f, list_s])
    label_list = label_list.sort_index()
    label_idx = label_list.index.values
    train_label = label_list.label.values
    x_train = x_train[label_idx]
    y_train= train_label
    y_train = np.array([[0, 1] if i==f_label else [1, 0] for i in y_train])

    df = pd.DataFrame(columns=["label"])
    df["label"] = y_test.reshape([-1])
    list_f = df.loc[df.label==f_label].sample(n=test_num)
    list_s = df.loc[df.label==s_label].sample(n=test_num)
    label_list = pd.concat([list_f, list_s])
    label_list = label_list.sort_index()
    label_idx = label_list.index.values
    test_label = label_list.label.values
    x_test = x_test[label_idx]
    y_test= test_label
    y_test = np.array([[0, 1] if i==f_label else [1, 0] for i in y_test])

    x_train, x_test = x_train * np.pi / 255.0, x_test * np.pi / 255.0
    x_train = x_train.reshape([train_num*2, 28, 28, 1])
    x_train = tf.image.resize(x_train, [height, width])
    x_train = np.array(x_train).reshape([train_num*2, height*width])

    x_test = x_test.reshape([test_num*2, 28, 28, 1])
    x_test = tf.image.resize(x_test, [height,width])
    x_test = np.array(x_test).reshape([test_num*2, height*width])

    return x_train, y_train, x_test, y_test


xtrain, ytrain, xtest, ytest = generate_binary_mnist(0, 1, 100, 20, 8, 8)

qxtrain = qtensornetwork.util.dtoq_miles(xtrain)
qxtest = qtensornetwork.util.dtoq_miles(xtest)

```

Preparing datasets.  In order to encode classical data
into quantum state, `xtrain` and `xtest` is converted into
`qxtrain` and `qxtest` using 
`qtensornetwork.util.dtoq_miles`. This function converts x to
(cos pi\*x/2, sin pi\*x/2).

```
qnum = 64
circuit = qtensornetwork.circuit.Circuit(qnum)

for i in range(qnum):
    circuit.set_init_state(qtensornetwork.components.State([i], None, train_idx=i))

```

Preparing quantum circuit by `qtensornetwork.circuit.Circuit`.

In order to set the quantum state of datasets,
method `set_init_state` and object `State` is called.
`State` has 3 arguments,

* `input_qubits`: list of number. The support of quantum state.

* `tensor`: ndarray. The tensor of quantum state.

* `train_idx`: number. The train index of qxtrain.

```

def complex_gate():
    Rz0 = RZ(0,0)
    Ry1 = RY(0,0)
    Rz2 = RZ(0,0)
    Rz3 = RZ(1,0)
    Ry4 = RY(1,0)
    Ry5 = RZ(1,0)
    CNOT6 = CNOT([0,1])
    U_gate = combine_gates([Rz0, Ry1, Rz2, Rz3, Ry4, Ry5, CNOT6])
    return U_gate

cgate = complex_gate()

```

Preparing gate. In `QTensorNet`, gate is implemented
as object `Gate`. `Gate` has 6 arguments,

* `input_qubits`: list of number. The support of gate.

* `params`: ndarray. The parameters of gate.

* `func`: function. The function from params to unitary tensor.

* `tensor`: ndarray. The unitary tensor of gate.

* `is_updated`: bool. `True` means to update parameters during optimization.

* `train_idx`: list of number. The train index of xtrain. See example 2 below.

In `qtensornetwork.gate`, several
default gates is prepared. And you can prepare
any unitary gate using argument `tensor` or
`func`, `params`.

Function `combine_gates` combines some gates and return
new gate.

```
layer = qtensornetwork.ansatz.TTN([i for i in range(qnum)], gate_input_num=2, gate_output_num=1, gate_func=cgate.func, gate_params_num=6)
circuit.append_layer(layer)
```

64 qubits quantum circuit cannot be simulated efficiently.
However, by using Tensor Network structure, the expectation
value of quantum circuit can be calculated in poly time.

Several ansatz is prepared in `qtensornetwork.ansatz`.

Tree Tensor Network structure `TTN` has 6 arguments,

* `q_support`: list of number. The support of tensor network structure.

* `is_updated`: bool. `True` means to update parameters during optimization.

* `gate_input_num`, `gate_output_num`: number. This decides the shape of each small gates.

* `gate_func`: function. The function of each small gates.

* `gate_params_num`: number. The number of parameters of each small gates.

After preparing TTN-structured layer, it can 
be appended to the quantum circuit using `circuit.append_layer`.

```
m_tensor = np.array([[1, 0], [0, 0]])
measurement1 = qtensornetwork.components.Measurement(None, m_tensor)
circuit.add_measurement(measurement1)

```
Add measurement. Object `Measurement` has 2 arguments,

* `input_qubits`: list of number. The support of measurement.`

* `tensor`: ndarray. The tensor of measurement, and it must be
hermitian.

If you want to use `classify` below, measurement tensor
must be POVM (in order to satisfy the condition the summation of 
output equals to 1).
```
circuit.show_circuit_structure()

optimizer = qtensornetwork.optimizer.Adam(lr=0.01)

circuit.classify(qxtrain, None, ytrain, qxtest, None, ytest, optimizer=optimizer, epoch=50, batch_size=20)
```

Optimizing procedure. `circuit.classify` for classification and 
 `circuit.fit` for fitting can be used.

# Example2
This example classifies binary `iris dataset`.

```
import jax.numpy as np
import numpy as onp
from qtensornetwork.circuit import Circuit
from qtensornetwork.components import Gate, Measurement
from qtensornetwork.util import data_to_qubits
from qtensornetwork.gate import *
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

onp.random.seed(0)

def generate_iris_binary_dataset():
    iris_dataset = load_iris()
    xtrain, xtest, ytrain, ytest = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
    mm = preprocessing.MinMaxScaler()
    xtrain = mm.fit_transform(xtrain)
    xtest = mm.fit_transform(xtest)

    xtr, ytr, xte, yte = [], [], [], []

    for i in range(len(xtrain)):
        if ytrain[i] == 0:
            xtr.append(xtrain[i])
            ytr.append([1, 0])
        if ytrain[i] == 1:
            xtr.append(xtrain[i])
            ytr.append([0, 1])

    for i in range(len(xtest)):
        if ytest[i] == 0:
            xte.append(xtest[i])
            yte.append([1, 0])
        if ytest[i] == 1:
            xte.append(xtest[i])
            yte.append([0, 1])

    xtrain = np.array(xtr)
    xtest = np.array(xte)
    ytrain = np.array(ytr)
    ytest = np.array(yte)

    return xtrain, ytrain, xtest, ytest

xtrain, ytrain, xtest, ytest = generate_iris_binary_dataset()

circuit = Circuit(4)

RYgate = RY(0,0)
for q in range(4):
    circuit.add_gate(Gate([q], params=None, func=RYgate.func, train_idx=[q]))

n = 3

def circuit1():
    gates = []
    gates.append(RX(0,onp.random.randn()))
    gates.append(RX(1,onp.random.randn()))
    gates.append(RX(2,onp.random.randn()))
    gates.append(RX(3,onp.random.randn()))
    gates.append(RZ(0,onp.random.randn()))
    gates.append(RZ(1,onp.random.randn()))
    gates.append(RZ(2,onp.random.randn()))
    gates.append(RZ(3,onp.random.randn()))
    gates.append(CNOT([3,2]))
    gates.append(CNOT([2,1]))
    gates.append(CNOT([1,0]))
    gate = combine_gates(gates)
    return gate

for i in range(n):
    gate = circuit1()
    gate.is_updated = True
    circuit.add_gate(gate)

m = np.array([[1, 0],[0,0]])
measurement = Measurement([0], m)
circuit.add_measurement(measurement)

circuit.show_circuit_structure()

circuit.classify(None, xtrain, ytrain, None, xtest, ytest, optimizer="adam", epoch=10)
```

# Example3
This example also clasifies binary `iris dataset`, but
using `MPS` ansatz for the circuit. `MPS` ansatz can be simulated
faster using tensor network.
```
import jax.numpy as np
import numpy as onp
from qtensornetwork.circuit import Circuit
from qtensornetwork.components import Gate, Measurement
from qtensornetwork.ansatz import MPS
from qtensornetwork.util import data_to_qubits
from qtensornetwork.gate import *
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

onp.random.seed(0)

def generate_iris_binary_dataset():
    iris_dataset = load_iris()
    xtrain, xtest, ytrain, ytest = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
    mm = preprocessing.MinMaxScaler()
    xtrain = mm.fit_transform(xtrain)
    xtest = mm.fit_transform(xtest)

    xtr, ytr, xte, yte = [], [], [], []

    for i in range(len(xtrain)):
        if ytrain[i] == 0:
            xtr.append(xtrain[i])
            ytr.append([1, 0])
        if ytrain[i] == 1:
            xtr.append(xtrain[i])
            ytr.append([0, 1])

    for i in range(len(xtest)):
        if ytest[i] == 0:
            xte.append(xtest[i])
            yte.append([1, 0])
        if ytest[i] == 1:
            xte.append(xtest[i])
            yte.append([0, 1])

    xtrain = np.array(xtr)
    xtest = np.array(xte)
    ytrain = np.array(ytr)
    ytest = np.array(yte)

    return xtrain, ytrain, xtest, ytest

xtrain, ytrain, xtest, ytest = generate_iris_binary_dataset()

circuit = Circuit(4)

RYgate = RY(0,0)
for q in range(4):
    circuit.add_gate(Gate([q], params=None, func=RYgate.func, train_idx=[q]))

layer = layer = MPS([i for i in range(4)], gate_input_num=2, gate_output_num=1)
circuit.append_layer(layer)

m = np.array([[1, 0],[0,0]])
measurement = Measurement(None, m)
circuit.add_measurement(measurement)

circuit.show_circuit_structure()

circuit.classify(None, xtrain, ytrain, None, xtest, ytest, optimizer="adam", epoch=10)
```

# Acknowledgements
This product is supported by [MITOU Target](https://www.ipa.go.jp/jinzai/target/2020/gaiyou_fk-1.html)．