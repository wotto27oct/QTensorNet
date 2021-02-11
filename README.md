# QTensorNet : Tensor-network backend Quantum Circuit Simulator

**QTensorNet** is Quantum Circuit & Quantum Machine Learning Simulator using Tensor Network.

We adopt **tensor-network backend**, so for some specific circuits, QTensorNet can calculate the expectation of the obserbable faster than other simulators.

Moreover, QTensorNet is specialized in Quantum Circuit Learning. You can try Quantum machine learning without any knowledge about quantum mechanics and tensor network.

**Note: This product is yet under develepment (alpha version). We don't guantee any calculation.**

# install
Download folder `qtensornetwork`.

# requirements
Python library [`jax, jaxlib`](https://github.com/google/jax), [`opt_einsum`](https://github.com/dgasmith/opt_einsum) is needed.

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


qnum = 4
circuit = qtensornetwork.circuit.Circuit(qnum)

xtrain, ytrain, xtest, ytest = generate_binary_mnist(0, 1, 10, 2, 2, 2)

qxtrain = qtensornetwork.util.dtoq_miles(xtrain)
qxtest = qtensornetwork.util.dtoq_miles(xtest)

for i in range(qnum):
    circuit.set_init_state(qtensornetwork.components.State([i], None, train_idx=i))

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
layer = qtensornetwork.ansatz.TTN([i for i in range(qnum)], gate_input_num=2, gate_output_num=1, gate_func=cgate.func, gate_params_num=6)
circuit.append_layer(layer)

m_tensor = np.array([[1, 0], [0, 0]])
measurement1 = qtensornetwork.components.Measurement(None, m_tensor)
circuit.add_measurement(measurement1)

circuit.show_circuit_structure()

optimizer = qtensornetwork.optimizer.Adam(lr=0.01)

print(ytrain.shape)

circuit.classify(qxtrain, None, ytrain, qxtest, None, ytest, optimizer=optimizer, epoch=100, batch_size=2)
```

# Example2
This example classifies binary `iris dataset`.

```
import jax.numpy as np
import numpy as onp
import qtensornetwork.components as qtc
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
        if ytrain[i] != 2:
            xtr.append(xtrain[i])
            ytr.append(ytrain[i])

    for i in range(len(xtest)):
        if ytest[i] != 2:
            xte.append(xtest[i])
            yte.append(ytest[i])
    xtrain = np.array(xtr)
    xtest = np.array(xte)
    ytrain = np.array(ytr)
    ytest = np.array(yte)

    qxtrain = data_to_qubits(xtrain, type="Miles")
    qxtest = data_to_qubits(xtest, type="Miles")
    return xtrain, qxtrain, ytrain, xtest, qxtest, ytest

xtrain, qxtrain, ytrain, xtest, qxtest, ytest = generate_iris_binary_dataset()

circuit = qtc.Circuit(4)

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
measurement = qtc.Measurement([0], m)
circuit.add_measurement(measurement)

circuit.show_circuit_structure()

circuit.fit(qxtrain, ytrain, qxtest, ytest, optimizer="adam", epoch=10)
```

# Example3
This example also clasifies binary `iris dataset`, but
using `MPS` ansatz for the circuit. `MPS` ansatz can be simulated
faster using tensor network.
```
import jax.numpy as np
import numpy as onp
import qtensornetwork.components as qtc
import qtensornetwork.ansatz as qta
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
        if ytrain[i] != 2:
            xtr.append(xtrain[i])
            ytr.append(ytrain[i])

    for i in range(len(xtest)):
        if ytest[i] != 2:
            xte.append(xtest[i])
            yte.append(ytest[i])
    xtrain = np.array(xtr)
    xtest = np.array(xte)
    ytrain = np.array(ytr)
    ytest = np.array(yte)

    qxtrain = data_to_qubits(xtrain, type="Miles")
    qxtest = data_to_qubits(xtest, type="Miles")
    return xtrain, qxtrain, ytrain, xtest, qxtest, ytest

xtrain, qxtrain, ytrain, xtest, qxtest, ytest = generate_iris_binary_dataset()

circuit = qtc.Circuit(4)

layer = layer = qta.MPS([i for i in range(4)], gate_input_num=2, gate_output_num=1)
circuit.append_layer(layer)

m = np.array([[1, 0],[0,0]])
measurement = qtc.Measurement(None, m)
circuit.add_measurement(measurement)

circuit.show_circuit_structure()

circuit.fit(qxtrain, ytrain, qxtest, ytest, optimizer="adam", epoch=10)
```

# Acknowledgements
This product is supported by [MITOU Target](https://www.ipa.go.jp/jinzai/target/2020/gaiyou_fk-1.html)．