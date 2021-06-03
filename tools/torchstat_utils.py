# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from torchstat import analyze
import pandas as pd
import copy


class LayerStats:

    def __init__(self, node) -> None:
        self.name = node.name
        self.input_shape = node.input_shape
        self.output_shape = node.output_shape
        self.parameters = node.parameter_quantity
        self.inference_memory = node.inference_memory
        self.MAdd = node.MAdd
        self.Flops = node.Flops
        self.mread, self.mwrite = node.Memory[0], node.Memory[1]
        self.duration = node.duration


class ModelStats(LayerStats):

    def __init__(self, model, input_shape, clone_model=False) -> None:
        if clone_model:
            model = copy.deepcopy(model)
        collected_nodes = analyze(model, input_shape, 1)
        self.layer_stats = []
        for node in collected_nodes:
            self.layer_stats.append(LayerStats(node))

        self.name = 'Model'
        self.input_shape = input_shape
        self.output_shape = self.layer_stats[-1].output_shape
        self.parameters = sum((l.parameters for l in self.layer_stats))
        self.inference_memory = sum(
            (l.inference_memory for l in self.layer_stats))
        self.MAdd = sum((l.MAdd for l in self.layer_stats))
        self.Flops = sum((l.Flops for l in self.layer_stats))
        self.mread = sum((l.mread for l in self.layer_stats))
        self.mwrite = sum((l.mwrite for l in self.layer_stats))
        self.duration = sum((l.duration for l in self.layer_stats))


def model_stats(model, input_shape):
    ms = ModelStats(model, input_shape)
    return model_stats2df(ms)


def _round_value(value, binary=False):
    divisor = 1024. if binary else 1000.

    if value // divisor**4 > 0:
        return str(round(value / divisor**4, 2)) + 'T'
    elif value // divisor**3 > 0:
        return str(round(value / divisor**3, 2)) + 'G'
    elif value // divisor**2 > 0:
        return str(round(value / divisor**2, 2)) + 'M'
    elif value // divisor > 0:
        return str(round(value / divisor, 2)) + 'K'
    return str(value)


def model_stats2df(model_stats: ModelStats):
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_rows', 10000)
    pd.set_option('display.max_columns', 10000)

    df = pd.DataFrame([l.__dict__ for l in model_stats.layer_stats])
    total_df = pd.Series(model_stats.__dict__, name='Total')
    df = df.append(total_df[df.columns], ignore_index=True)

    df = df.fillna(' ')
    # df['memory(MB)'] = df['memory(MB)'].apply(
    #     lambda x: '{:.2f}'.format(x))
    # df['duration[%]'] = df['duration[%]'].apply(lambda x: '{:.2%}'.format(x))
    for c in [
            'MAdd', 'Flops', 'parameters', 'inference_memory', 'mread',
            'mwrite'
    ]:  
        if c == 'Flops':
            df[c] = df[c].apply(lambda x: _round_value(x, True))
        elif c == 'parameters':
            df[c] = df[c].apply(lambda x: _round_value(x))
        else:
            df[c] = df[c].apply(lambda x: '{:,}'.format(x))

    df.rename(
        columns={
            'name': 'module name',
            'input_shape': 'input shape',
            'input_shape': 'input shape',
            'inference_memory': 'infer memory(MB)',
            'mread': 'MemRead(B)',
            'mwrite': 'MemWrite(B)'
        },
        inplace=True)

    #summary = "Total params: {:,}\n".format(total_parameters_quantity)

    #summary += "-" * len(str(df).split('\n')[0])
    #summary += '\n'
    #summary += "Total memory: {:.2f}MB\n".format(total_memory)
    #summary += "Total MAdd: {}MAdd\n".format(_round_value(total_operation_quantity))
    #summary += "Total Flops: {}Flops\n".format(_round_value(total_flops))
    #summary += "Total MemR+W: {}B\n".format(_round_value(total_memrw, True))
    return df