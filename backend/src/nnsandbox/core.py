import numpy as np
from collections import defaultdict

NP_EMPTY = np.array(0)
SUM_AX_0 = lambda x: np.sum(x, 0)
# vjp of SUM_AX_0: s = c_1 + ... + c_k, so dL/dc_i = dL/ds for every i.
D_SUM_AX_0 = lambda contribs, g: [g] * len(contribs)

class Layer:
    def __init__(self, activation, d_activation, compounder=SUM_AX_0, d_compounder=D_SUM_AX_0,
                 size: int=0, from_array: np.ndarray=NP_EMPTY, bias: np.ndarray=NP_EMPTY) -> None:
        if len(from_array.shape) == 1:
            self.values = from_array.copy().reshape(1, -1)
        elif size <= 0:
            raise ValueError("Layer must be pre-populated with a flat list / array or size must be positive")
        else:
            self.values = np.random.rand(1, size) - 0.5

        bias_shape = (1, self.size())
        if bias.shape:
            if bias.shape != bias_shape:
                raise ValueError(f"bias dimension does not match {bias_shape}: {bias.shape}")
            self.bias = bias.copy()
        else:
            self.bias = np.random.rand(*bias_shape) - 0.5

        # activation / compounder are the forward ops; d_* are their local gradients.
        #   d_activation(pre_activation) -> elementwise derivative, same shape
        #   d_compounder(contribs, g)    -> list of dL/dc_i given g = dL/d(compounder output)
        self.activation = activation
        self.d_activation = d_activation
        self.compounder = compounder
        self.d_compounder = d_compounder

        self.pre_activation = NP_EMPTY  # cached z = bias + compounder(...), set by compute()
        self.from_links: list["Link"] = []
        self.to_links: list["Link"] = []


    def compute(self) -> np.ndarray:
        '''Recompute this layer's values from its already-computed predecessors.

        Assumes every predecessor has been computed earlier this pass -- see
        Network.run, which walks Network.topo_order. Input layers have no
        from_links and keep whatever values they were configured with.
        '''
        if not self.from_links:
            return self.values
        self.pre_activation = self.bias + self.compounder(
            [link.from_layer.values @ link.weights for link in self.from_links]
        )
        self.values = self.activation(self.pre_activation)
        return self.values


    def size(self):
        return self.values.shape[1]


    def connect_to(self, other: "Layer", weights: np.ndarray=NP_EMPTY) -> "Link":
        '''Build a weighted link from this layer to `other`, registered on both.'''
        return Link(self, other, weights)


    def __str__(self) -> str:
        return f"Layer of size {self.size()}"


class Link:
    def __init__(self, from_layer: Layer, to_layer: Layer, weights: np.ndarray=NP_EMPTY) -> None:
        weights_shape = (from_layer.size(), to_layer.size())

        if weights.shape:
            if weights.shape != weights_shape:
                raise ValueError(f"weights dimension does not match {weights_shape}: {weights.shape}")
            self.weights = weights.copy()
        else:
            self.weights = np.random.rand(*weights_shape) - 0.5

        self.from_layer = from_layer
        self.to_layer = to_layer

        # A link lives on the two layers it joins rather than in a global list.
        from_layer.to_links.append(self)
        to_layer.from_links.append(self)


    def __str__(self) -> str:
        return f"Link of size {self.weights.shape}"


class Network:
    def __init__(self, cost, d_cost, layers: list[Layer]=[]) -> None:
        self.layers = list(layers) if layers else []
        self.in_adj = defaultdict(set)
        self.out_adj = defaultdict(set)
        self.inputs, self.outputs = self.calculate_io()
        self.cost, self.d_cost = cost, d_cost


    @property
    def links(self) -> list[Link]:
        seen = set()
        result = []
        for layer in self.layers:
            for link in layer.to_links:
                if id(link) not in seen:
                    seen.add(id(link))
                    result.append(link)
        return result


    def calculate_io(self) -> tuple[list[Layer], list[Layer]]:
        self.in_adj.clear()
        self.out_adj.clear()
        i, o = set(self.layers), set(self.layers)
        for layer in self.layers:
            for link in layer.to_links:
                self.in_adj[link.to_layer].add(link.from_layer)
                self.out_adj[link.from_layer].add(link.to_layer)
                o.discard(link.from_layer)
                i.discard(link.to_layer)

        return list(i), list(o)


    def topo_order(self) -> list[Layer]:
        '''Layers in dependency order: each appears after all its predecessors.

        Shared by the forward pass (this order) and the backward pass
        (reversed). Derived from each layer's own from_links / to_links, which
        Link.__init__ keeps current, so it never depends on stale adjacency.
        '''
        layer_set = set(self.layers)
        indeg = {
            layer: sum(link.from_layer in layer_set for link in layer.from_links)
            for layer in self.layers
        }
        queue = [layer for layer in self.layers if indeg[layer] == 0]
        order = []
        while queue:
            layer = queue.pop()
            order.append(layer)
            for link in layer.to_links:
                succ = link.to_layer
                if succ not in indeg:
                    continue
                indeg[succ] -= 1
                if indeg[succ] == 0:
                    queue.append(succ)

        if len(order) != len(self.layers):
            raise ValueError("Network has a cycle; cannot order layers")
        return order


    def add_layer(self, layer: Layer) -> None:
        if layer not in self.layers:
            self.layers.append(layer)
            self.inputs, self.outputs = self.calculate_io()


    def connect(self, from_layer: Layer, to_layer: Layer, weights: np.ndarray=NP_EMPTY) -> Link:
        link = from_layer.connect_to(to_layer, weights)
        self.inputs, self.outputs = self.calculate_io()
        return link


    def run(self) -> list[np.ndarray]:
        print('Network.run() assumes input values are configured before function call')

        for layer in self.topo_order():
            layer.compute()

        return [o.values for o in self.outputs]

    def backprop(self, input: dict[Layer, np.ndarray], real_outputs: dict[Layer, np.ndarray]):
        '''Forward once, then reverse-topo backward. Returns (cost, grad_weights,
        grad_bias); it computes gradients, it does not apply them.

        d_cost mirrors cost: d_cost(preds, targets) -> list of dL/da, one per
        output layer, each shaped like that layer's values.
        '''
        for l in self.inputs:
            if l not in input:
                raise ValueError(f"input key not found:\t{l}")
            l.values = input[l]
        for l in self.outputs:
            if l not in real_outputs:
                raise ValueError(f"target key not found:\t{l}")

        order = self.topo_order()
        for layer in order:
            layer.compute()

        preds = [l.values for l in self.outputs]
        targets = [real_outputs[l] for l in self.outputs]
        cost = self.cost(preds, targets)

        # grad_a[L] = dL/d(L.values), accumulated from every successor of L (and,
        # for output layers, straight from the cost). All rows, matching forward.
        grad_a = {layer: np.zeros((1, layer.size())) for layer in self.layers}
        for out_layer, g in zip(self.outputs, self.d_cost(preds, targets)):
            grad_a[out_layer] = grad_a[out_layer] + g

        grad_weights: dict[Link, np.ndarray] = {}
        grad_bias: dict[Layer, np.ndarray] = {}

        # Reverse topo order => every successor of `layer` is already done, so
        # grad_a[layer] is fully summed before we use it.
        for layer in reversed(order):
            if not layer.from_links:
                continue  # input layer: nothing feeds it, nothing to fit

            # dL/da -> dL/dz through the (elementwise) activation. This is delta_l.
            delta = grad_a[layer] * layer.d_activation(layer.pre_activation)  # (1, n_layer)
            grad_bias[layer] = delta.sum(axis=0, keepdims=True)              # z = bias + s

            # dL/dz == dL/ds; split it across the incoming contributions via the
            # compounder's vjp, then through each link's weights.
            contribs = [link.from_layer.values @ link.weights for link in layer.from_links]
            for link, d_contrib in zip(layer.from_links, layer.d_compounder(contribs, delta)):
                src = link.from_layer
                grad_weights[link] = src.values.T @ d_contrib             # (n_src, n_layer)
                grad_a[src] = grad_a[src] + d_contrib @ link.weights.T    # hand dL/da back

        return cost, grad_weights, grad_bias

    def train(self, train_data: list[dict[Layer, np.ndarray]], train_labels: list[dict[Layer, np.ndarray]], epochs: int):
        for e in range(epochs):
            tc = 0
            for input, label in zip(train_data, train_labels):
                cost, grad_weights, grad_bias = self.backprop(input, label)
                tc += cost
                for link, delta in grad_weights.items():
                    link.weights -= delta
                for layer, delta in grad_bias.items():
                    layer.bias -= delta
            print(f"Average cost for epoch {e+1}: {tc/len(train_data)}")

