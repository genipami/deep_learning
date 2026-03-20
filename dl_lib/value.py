import graphviz

class Value():
    def __init__(self, data, _prev=None, _op = None):
        self.data = data
        self._prev = _prev
        self._op = _op


    def __str__(self):
        return f"Value(data={self.data})"
    
    def __repr__(self):
        return f"Value(data={self.data})"


    def __add__(self, other):
        return Value(self.data + other.data, {self, other}, "+")
    
    def __mul__(self, other):
        return Value(self.data * other.data, {self, other}, "*")
    
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
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node
        dot.node(name=uid, label=f'{{ data: {n.data} }}', shape='record')
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
    x = Value(2.0)
    y = Value(-3.0)
    z = Value(10.0)
    result = x * y + z
    
    # This will create a new directory and store the output file there.
    # With "view=True" it'll automatically display the saved file.
    draw_dot(result).render(directory='./graphviz_output', view=True)

if __name__ == "__main__":
    main()    