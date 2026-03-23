import graphviz

class Value():
    def __init__(self, data:int, _prev:list = None, _op:str = None, label:str = None, gradient = 0):
        self.data = data
        self._prev = _prev
        self._op = _op
        self.label = label
        self.gradient = gradient


    def __str__(self):
        return f"Value(data={self.data})"
    
    def __repr__(self):
        return f"Value(data={self.data})"


    def __add__(self, other):
        return Value(self.data + other.data, {self, other}, "+")
    
    def __mul__(self, other):
        return Value(self.data * other.data, {self, other}, "*")
    
    def set_label(self, label:str):
        if label is not None:
            self.label = label

    def set_gradient(self, gradient:float):
        if gradient is not None:
            self.gradient = gradient
    
def trace(input):
        queue = [input]
        nodes = set()
        edges = set()

        while len(queue) > 0:
            curr = queue.pop(0)
            nodes.add(curr)

            if curr._prev is None:
                continue
            
            for prev in curr._prev:
                edge = (prev, curr)
                edges.add(edge)
                queue.append(prev)

        return(nodes, edges)    

def draw_dot(root: Value) -> graphviz.Digraph:
    dot = graphviz.Digraph(filename='01_result', format='svg', graph_attr={
                           'rankdir': 'LR'})  # LR = left to right

    nodes, edges = trace(root)
    for i, n in enumerate(nodes):
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node
        dot.node(name=uid, label=f'{{ {n.label} | data: {n.data} | grad: {n.gradient}}}', shape='record')
        if n._op:
            # if this value is a result of some operation, create an "op" node for the operation
            dot.node(name=uid + n._op, label=n._op)
            # and connect this node to the node of the operation
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # connect n1 to the "op" node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot


def main() -> None:
    x = Value(2.0, label = "a")
    y = Value(-3.0, label = "b")
    z = Value(10.0, label = "c")
    a = Value(5.0, label = "f")
    x_times_y = x * y
    x_times_y.set_label("e")
    x_times_y_plus_z = x_times_y + z
    x_times_y_plus_z.set_label("d")
    result:Value = x_times_y_plus_z * a
    result.set_label("L")
    result.set_gradient(1.0) #the derivative of L in regards to L is 1
    a.set_gradient(4.0) #the derivative of L in regards to f is (d*f)' => d = 4 
    x_times_y_plus_z.set_gradient(5.0) #similarly to the previous one the derivative of L in regards to d is (d*f)' => f = 5
    z.set_gradient(5.0) #the derivative of L in regards to c is (the derivative of L in regards to d)*(the derivative of d in regards to c) => 5*1 => 5
    x_times_y.set_gradient(5.0) #the derivative of L in regards to e is (the derivative of L in regards to d)*(the derivative of d in regards to e) => 5*1 => 5
    x.set_gradient(-15.0) #the derivative of L in regards to a is (the derivative of L in regards to e)*(the derivative of e in regards to a) => 5*(-3) => -15
    y.set_gradient(10.0) #the derivative of L in regards to b is (the derivative of L in regards to e)*(the derivative of e in regards to b) => 5*2 => 10
    
    # This will create a new directory and store the output file there.
    # With "view=True" it'll automatically display the saved file.
    draw_dot(result).render(directory='./graphviz_output', view=True)

if __name__ == "__main__":
    main()    