import pandas as pd

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 10000)


def round_value(value, binary=False):
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


def report_format(collected_nodes):
    data = list()
    for node in collected_nodes:
        name = node.name
        input_shape = ' '.join(
            ['{:>3d}'] *
            len(node.input_shape)).format(*[e for e in node.input_shape])
        output_shape = ' '.join(
            ['{:>3d}'] *
            len(node.output_shape)).format(*[e for e in node.output_shape])
        parameter_quantity = node.parameter_quantity
        inference_memory = node.inference_memory
        MAdd = node.MAdd
        Flops = node.Flops
        mread, mwrite = [i for i in node.Memory]
        duration = node.duration
        data.append([
            name, input_shape, output_shape, parameter_quantity,
            inference_memory, MAdd, duration, Flops, mread, mwrite
        ])
    df = pd.DataFrame(data)
    df.columns = [
        'module name', 'input shape', 'output shape', 'params', 'memory(MB)',
        'MAdd', 'duration', 'Flops', 'MemRead(B)', 'MemWrite(B)'
    ]
    df['duration[%]'] = df['duration'] / (df['duration'].sum() + 1e-7)
    df['MemR+W(B)'] = df['MemRead(B)'] + df['MemWrite(B)']
    total_parameters_quantity = df['params'].sum()
    total_memory = df['memory(MB)'].sum()
    total_operation_quantity = df['MAdd'].sum()
    total_flops = df['Flops'].sum()
    total_duration = df['duration[%]'].sum()
    total_mread = df['MemRead(B)'].sum()
    total_mwrite = df['MemWrite(B)'].sum()
    total_memrw = df['MemR+W(B)'].sum()
    del df['duration']

    # Add Total row
    total_df = pd.Series([
        total_parameters_quantity, total_memory, total_operation_quantity,
        total_flops, total_duration, mread, mwrite, total_memrw
    ],
                         index=[
                             'params', 'memory(MB)', 'MAdd', 'Flops',
                             'duration[%]', 'MemRead(B)', 'MemWrite(B)',
                             'MemR+W(B)'
                         ],
                         name='total')
    df = df.append(total_df)

    df = df.fillna(' ')
    df['memory(MB)'] = df['memory(MB)'].apply(lambda x: '{:.2f}'.format(x))
    df['duration[%]'] = df['duration[%]'].apply(lambda x: '{:.2%}'.format(x))
    df['MAdd'] = df['MAdd'].apply(lambda x: '{:,}'.format(x))
    df['Flops'] = df['Flops'].apply(lambda x: '{:,}'.format(x))

    summary = str(df) + '\n'
    summary += "=" * len(str(df).split('\n')[0])
    summary += '\n'
    summary += "Total params: {:,}\n".format(total_parameters_quantity)

    summary += "-" * len(str(df).split('\n')[0])
    summary += '\n'
    summary += "Total memory: {:.2f}MB\n".format(total_memory)
    summary += "Total MAdd: {}MAdd\n".format(
        round_value(total_operation_quantity))
    summary += "Total Flops: {}Flops\n".format(round_value(total_flops))
    summary += "Total MemR+W: {}B\n".format(round_value(total_memrw, True))
    return summary
