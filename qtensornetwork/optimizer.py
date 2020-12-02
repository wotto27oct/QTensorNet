import jax.numpy as np
import numpy as onp
import math
import time

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.name = "sgd"
    
    def update(self, params, grads):
        for i in range(len(params)):
            if params[i] is not None:
                params[i] -= self.lr * grads[i]
    
    def update_with_index(self, params, grads, index):
        if params[index] is not None:
            params[index] -= self.lr * grads[index]

class ManSGD:
    def __init__(self, lr=0.01, ret_type="QR"):
        self._lr = lr
        self.tau = 0.001
        self.rho = 0.5
        self.ret_type = ret_type
        self.name = "mansgd"

    def simple_retract(self, w, X):
        ans = [0] * len(w)
        for i in range(len(w)):
            if w[i] is not None:
                ans[i] = (w[i] + X[i]) / np.linalg.norm(w[i] + X[i])
            else:
                ans[i] = None
        return ans

    def QR_retract(self, w, X):
        ret = [0] * len(w)
        for i in range(len(w)):
            if w[i] is not None:
                Q, R = np.linalg.qr(w[i] + X[i])
                S = onp.sign(onp.diag(R.diagonal()))
                ret[i] = np.dot(Q, S)
            else:
                ret[i] = None
        return ret
    
    def Polar_retract(self, w, X):
        ret = [0] * len(w)
        for i in range(len(w)):
            if w[i] is not None:
                U, _, Vh = np.linalg.svd(w[i] + X[i])
                ret[i] = np.dot(U, Vh)
            else:
                ret[i] = None
        return ret
    
    def update(self, params, f_params, gradf):
        alpha = self._lr
        D = gradf(params)
        sh = [0] * len(params)
        W = [0] * len(params)
        G = [0] * len(params)
        for i in range(len(params)):
            if params[i] is not None:
                sh[i] = int(math.sqrt(len(params[i])))
                D[i] = D[i].reshape(sh[i], sh[i])
                W[i] = params[i].reshape(sh[i], sh[i])
                G[i] = D[i] - np.dot(W[i], (np.dot(W[i].T, D[i]) + np.dot(D[i].T, W[i]))) / 2.0
            else:
                sh[i] = None
                D[i] = None
                W[i] = None
                G[i] = None
        fw = f_params(params)
        def f(W):
            par = [0] * len(params)
            for i in range(len(params)):
                if params[i] is not None:
                    par[i] = W[i].flatten()
                else:
                    par[i] = None
            return f_params(par)
        d = [-G[i] if G[i] is not None else None for i in range(len(params))]
        dD = 0
        for i in range(len(W)):
            if params[i] is not None:
                dD += np.trace(np.dot(d[i].T, D[i]))

        ret_func = None
        if self.ret_type == "QR":
            ret_func = self.QR_retract
        elif self.ret_type == "Polar":
            ret_func = self.Polar_retract

        while f(ret_func(W, [alpha * d[i] if d[i] is not None else None for i in range(len(W))])) > fw + self.tau*alpha*dD:
            alpha *= self.rho
            if alpha < 1e-15:
                break
        W = ret_func(W, [alpha * d[i] if d[i] is not None else None for i in range(len(W))])
        for i in range(len(params)):
            if W[i] is not None:
                params[i] = W[i].flatten()
        
        

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        self.name = "adam"

    def update(self, params, grads):
        if self.m is None:
            self.m = [0] * len(params)
            self.v = [0] * len(params)
            for i in range(len(params)):
                if params[i] is not None:
                    self.m[i] = np.zeros_like(params[i])
                    self.v[i] = np.zeros_like(params[i])
        
        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        for i in range(len(params)):
            if params[i] is not None:
                self.m[i] += (1 - self.beta1) * (grads[i] - self.m[i])
                self.v[i] += (1 - self.beta2) * (grads[i]**2 - self.v[i])
                params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-7)

    def update_with_index(self, params, grads, index):
        if params[index] is None:
            return
        
        if self.m is None:
            self.m = [0] * len(params)
            self.v = [0] * len(params)
            for i in range(len(params)):
                if params[i] is not None:
                    self.m[i] = np.zeros_like(params[i])
                    self.v[i] = np.zeros_like(params[i])
            self.iter = [0] * len(params)
        
        self.iter[index] += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter[index]) / (1.0 - self.beta1**self.iter[index])

        self.m[index] += (1 - self.beta1) * (grads[index] - self.m[index])
        self.v[index] += (1 - self.beta2) * (grads[index]**2 - self.v[index])
        params[index] -= lr_t * self.m[index] / (np.sqrt(self.v[index]) + 1e-7)

class Adam_NGD:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        self.name="adam_ngd"

    def update(self, params, grads):
        if self.m is None:
            self.m = [0] * len(params)
            self.v = [0] * len(params)
            for i in range(len(params)):
                if params[i] is not None:
                    self.m[i] = np.zeros_like(params[i])
                    self.v[i] = np.zeros_like(params[i])
        
        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        for i in range(len(params)):
            if params[i] is not None:
                self.m[i] += (1 - self.beta1) * (grads[i] - self.m[i])
                self.v[i] += (1 - self.beta2) * (grads[i]**2 - self.v[i])
                params[i] -= lr_t * self.m[i] / (self.v[i] + 1e-7)

class RAdam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        self.rho_inf = 2 / (1 - self.beta2) - 1
        self.name = "radam"

    def update(self, params, grads):
        if self.m is None:
            self.m = [0] * len(params)
            self.v = [0] * len(params)
            for i in range(len(params)):
                if params[i] is not None:
                    self.m[i] = np.zeros_like(params[i])
                    self.v[i] = np.zeros_like(params[i])
        
        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        rho_t = self.rho_inf - 2 * self.iter * (self.beta2**self.iter) / (1.0 - self.beta2**self.iter)

        for i in range(len(params)):
            if params[i] is not None:
                self.m[i] += (1 - self.beta1) * (grads[i] - self.m[i])
                self.v[i] += (1 - self.beta2) * (grads[i]**2 - self.v[i])
                if rho_t  <= 4:
                    params[i] -= self.lr * self.m[i] / (1.0 - self.beta1**self.iter)
                else:
                    rt = np.sqrt(((rho_t - 4.0) * (rho_t - 2.0) * self.rho_inf) / ((self.rho_inf - 4.0) * (self.rho_inf - 2.0) * rho_t))
                    params[i] -= lr_t * rt * self.m[i] / (np.sqrt(self.v[i]) + 1e-7)