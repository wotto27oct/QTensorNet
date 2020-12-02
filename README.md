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

# Example2
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