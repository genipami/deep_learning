import graphviz
from dl_lib.value import Value, trace
def draw_dot(root: Value) -> graphviz.Digraph:
    dot = graphviz.Digraph(filename='01_result', format='svg', graph_attr={
                           'rankdir': 'LR'})  # LR = left to right

    nodes, edges = trace(root)
    for i, n in enumerate(nodes):
        uid = str(id(n))
        label = chr(ord('a') + i)
        # for any value in the graph, create a rectangular ('record') node
        dot.node(name=uid, label=f'{{ {label} | data: {n.data} }}', shape='record')
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
    a = Value(5)
    result = (x * y + z) * a
    
    # This will create a new directory and store the output file there.
    # With "view=True" it'll automatically display the saved file.
    draw_dot(result).render(directory='./graphviz_output', view=True)

if __name__ == "__main__":
    main()    