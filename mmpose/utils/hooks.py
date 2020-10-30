import functools


class Hook:

    def __init__(self, module):
        self.module = module
        self.handles = None
        self.register(self.module)

    def register(self, module):
        pass

    def remove(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()


class OutputsHook(Hook):

    def __init__(self, module, outputs=None):
        self.outputs = outputs
        self.layer_outputs = {}
        super().__init__(module)

    def register(self, module):

        def hook_wrapper(name):

            def hook(model, input, output):
                self.layer_outputs[name] = output.detach().cpu().numpy()

            return hook

        self.handles = []
        if isinstance(self.outputs, (list, tuple)):
            for name in self.outputs:
                try:
                    layer = rgetattr(module, name)
                    h = layer.register_forward_hook(hook_wrapper(name))
                except AttributeError:
                    if name in ('heatmap', 'heatmaps'):
                        continue
                    else:
                        raise AttributeError(f'Module {name} not found')
                self.handles.append(h)

    def remove(self):
        for h in self.handles:
            h.remove()


# using wonder's beautiful simplification:
# https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects
def rgetattr(obj, attr, *args):

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))
