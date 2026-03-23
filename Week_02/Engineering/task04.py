class Value():
    def __init__(self, data, _prev=None):
        self.data = data
        self._prev = _prev

    def __str__(self):
        return f"Value(data={self.data})"
    
    def __repr__(self):
        return f"Value(data={self.data})"


    def __add__(self, other):
        return Value(self.data + other.data, {self, other})
    
    def __mul__(self, other):
        return Value(self.data * other.data, {self, other})
    