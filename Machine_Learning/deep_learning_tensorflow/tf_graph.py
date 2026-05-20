import numpy as np

class Tensor:
    def __init__(self, value=None, parents=None, op=None, name=None):
        self.value = value              
        self.grad = None                
        self.parents = parents or []    
        self.op = op                    
        self.name = name

    def zero_grad(self):
        self.grad = np.zeros_like(self.value)


class Layer:
    def __call__(self, *inputs):
        output = Tensor(parents=list(inputs), op=self)
        return output

    def forward(self, *inputs):
        raise NotImplementedError

    def backward(self, out_tensor):
        raise NotImplementedError

    @property
    def params(self):
        return []

class Dense(Layer):
    def __init__(self, in_dim, out_dim):
        self.W = np.random.randn(in_dim, out_dim) * 0.1
        self.b = np.zeros(out_dim)

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        return x @ self.W + self.b

    def backward(self, out):
        x = out.parents[0]

        self.dW += x.value.T @ out.grad
        self.db += out.grad.sum(axis=0)

        x.grad += out.grad @ self.W.T

    @property
    def params(self):
        return [(self.W, self.dW), (self.b, self.db)]

class Input(Layer):
    def __init__(self, shape, name=None):
        self.shape = shape
        self.name = name or 'input'

    def __call__(self):
        return Tensor(name=self.name)


class Flatten(Layer):
    def forward(self, x):
        self.orig_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, out):
        x = out.parents[0]
        x.grad += out.grad.reshape(self.orig_shape)

class Concatenate(Layer):
    def forward(self, *xs):
        self.sizes = [x.shape[self.axis] for x in xs]
        return np.concatenate(xs, axis=self.axis)

    def backward(self, out):
        start = 0
        for parent, size in zip(out.parents, self.sizes):
            slc = [slice(None)] * out.grad.ndim
            slc[self.axis] = slice(start, start + size)
            parent.grad += out.grad[tuple(slc)]
            start += size


def topo_sort(output):
    visited = set()
    order = []

    def dfs(t):
        if id(t) in visited:
            return
        visited.add(id(t))
        for p in t.parents:
            dfs(p)
        order.append(t)

    dfs(output)
    return order



class Model:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.graph = topo_sort(outputs)

        self.layers = []
        for t in self.graph:
            if t.op and t.op not in self.layers:
                self.layers.append(t.op)

    def forward(self, input_dict):
        # assign input values
        for inp in self.inputs:
            inp.value = input_dict[inp.name]

        for t in self.graph:
            if t.op:
                vals = [p.value for p in t.parents]
                t.value = t.op.forward(*vals)

        return self.outputs.value

    def backward(self):
        for t in self.graph:
            t.zero_grad()

        self.outputs.grad = np.ones_like(self.outputs.value)

        for t in reversed(self.graph):
            if t.op:
                t.op.backward(t)

    def step(self, lr=0.01):
        for layer in self.layers:
            for param, grad in layer.params:
                param -= lr * grad
