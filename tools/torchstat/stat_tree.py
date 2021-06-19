import queue


class StatTree(object):

    def __init__(self, root_node):
        assert isinstance(root_node, StatNode)

        self.root_node = root_node

    def get_same_level_max_node_depth(self, query_node):
        if query_node.name == self.root_node.name:
            return 0
        same_level_depth = max(
            [child.depth for child in query_node.parent.children])
        return same_level_depth

    def update_stat_nodes_granularity(self):
        q = queue.Queue()
        q.put(self.root_node)
        while not q.empty():
            node = q.get()
            node.granularity = self.get_same_level_max_node_depth(node)
            for child in node.children:
                q.put(child)

    def get_collected_stat_nodes(self, query_granularity):
        self.update_stat_nodes_granularity()

        collected_nodes = []
        stack = list()
        stack.append(self.root_node)
        while len(stack) > 0:
            node = stack.pop()
            for child in reversed(node.children):
                stack.append(child)
            if node.depth == query_granularity:
                collected_nodes.append(node)
            if node.depth < query_granularity <= node.granularity:
                collected_nodes.append(node)
        return collected_nodes


class StatNode(object):

    def __init__(self, name=str(), parent=None):
        self._name = name
        self._input_shape = None
        self._output_shape = None
        self._parameter_quantity = 0
        self._inference_memory = 0
        self._MAdd = 0
        self._Memory = (0, 0)
        self._Flops = 0
        self._duration = 0
        self._duration_percent = 0

        self._granularity = 1
        self._depth = 1
        self.parent = parent
        self.children = list()

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def granularity(self):
        return self._granularity

    @granularity.setter
    def granularity(self, g):
        self._granularity = g

    @property
    def depth(self):
        d = self._depth
        if len(self.children) > 0:
            d += max([child.depth for child in self.children])
        return d

    @property
    def input_shape(self):
        if len(self.children) == 0:  # leaf
            return self._input_shape
        else:
            return self.children[0].input_shape

    @input_shape.setter
    def input_shape(self, input_shape):
        assert isinstance(input_shape, (list, tuple))
        self._input_shape = input_shape

    @property
    def output_shape(self):
        if len(self.children) == 0:  # leaf
            return self._output_shape
        else:
            return self.children[-1].output_shape

    @output_shape.setter
    def output_shape(self, output_shape):
        assert isinstance(output_shape, (list, tuple))
        self._output_shape = output_shape

    @property
    def parameter_quantity(self):
        # return self.parameters_quantity
        total_parameter_quantity = self._parameter_quantity
        for child in self.children:
            total_parameter_quantity += child.parameter_quantity
        return total_parameter_quantity

    @parameter_quantity.setter
    def parameter_quantity(self, parameter_quantity):
        assert parameter_quantity >= 0
        self._parameter_quantity = parameter_quantity

    @property
    def inference_memory(self):
        total_inference_memory = self._inference_memory
        for child in self.children:
            total_inference_memory += child.inference_memory
        return total_inference_memory

    @inference_memory.setter
    def inference_memory(self, inference_memory):
        self._inference_memory = inference_memory

    @property
    def MAdd(self):
        total_MAdd = self._MAdd
        for child in self.children:
            total_MAdd += child.MAdd
        return total_MAdd

    @MAdd.setter
    def MAdd(self, MAdd):
        self._MAdd = MAdd

    @property
    def Flops(self):
        total_Flops = self._Flops
        for child in self.children:
            total_Flops += child.Flops
        return total_Flops

    @Flops.setter
    def Flops(self, Flops):
        self._Flops = Flops

    @property
    def Memory(self):
        total_Memory = self._Memory
        for child in self.children:
            total_Memory[0] += child.Memory[0]
            total_Memory[1] += child.Memory[1]
            print(total_Memory)
        return total_Memory

    @Memory.setter
    def Memory(self, Memory):
        assert isinstance(Memory, (list, tuple))
        self._Memory = Memory

    @property
    def duration(self):
        total_duration = self._duration
        for child in self.children:
            total_duration += child.duration
        return total_duration

    @duration.setter
    def duration(self, duration):
        self._duration = duration

    def find_child_index(self, child_name):
        assert isinstance(child_name, str)

        index = -1
        for i in range(len(self.children)):
            if child_name == self.children[i].name:
                index = i
        return index

    def add_child(self, node):
        assert isinstance(node, StatNode)

        if self.find_child_index(node.name) == -1:  # not exist
            self.children.append(node)
