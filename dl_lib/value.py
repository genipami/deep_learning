

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


