# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple, Union

from mmengine.config import Config, ConfigDict
from mmengine.dist import master_only
from mmengine.logging import MMLogger

ConfigType = Union[Config, ConfigDict]


def process_input_transform(input_transform: str, head: Dict, head_new: Dict,
                            head_deleted_dict: Dict, head_append_dict: Dict,
                            neck_new: Dict, input_index: Tuple[int],
                            align_corners: bool) -> None:
    """Process the input_transform field and update head and neck
    dictionaries."""
    if input_transform == 'resize_concat':
        in_channels = head_new.pop('in_channels')
        head_deleted_dict['in_channels'] = str(in_channels)
        in_channels = sum([in_channels[i] for i in input_index])
        head_new['in_channels'] = in_channels
        head_append_dict['in_channels'] = str(in_channels)

        neck_new.update(
            dict(
                type='FeatureMapProcessor',
                concat=True,
                select_index=input_index,
            ))
        if align_corners:
            neck_new['align_corners'] = align_corners

    elif input_transform == 'select':
        if input_index != (-1, ):
            neck_new.update(
                dict(type='FeatureMapProcessor', select_index=input_index))
            if isinstance(head['in_channels'], tuple):
                in_channels = head_new.pop('in_channels')
                head_deleted_dict['in_channels'] = str(in_channels)
                if isinstance(input_index, int):
                    in_channels = in_channels[input_index]
                else:
                    in_channels = tuple([in_channels[i] for i in input_index])
                head_new['in_channels'] = in_channels
                head_append_dict['in_channels'] = str(in_channels)
            if align_corners:
                neck_new['align_corners'] = align_corners

    else:
        raise ValueError(f'model.head get invalid value for argument '
                         f'input_transform: {input_transform}')


def process_extra_field(extra: Dict, head_new: Dict, head_deleted_dict: Dict,
                        head_append_dict: Dict, neck_new: Dict) -> None:
    """Process the extra field and update head and neck dictionaries."""
    head_deleted_dict['extra'] = 'dict('
    for key, value in extra.items():
        head_deleted_dict['extra'] += f'{key}={value},'
    head_deleted_dict['extra'] = head_deleted_dict['extra'][:-1] + ')'
    if 'final_conv_kernel' in extra:
        kernel_size = extra['final_conv_kernel']
        if kernel_size > 1:
            padding = kernel_size // 2
            head_new['final_layer'] = dict(
                kernel_size=kernel_size, padding=padding)
            head_append_dict[
                'final_layer'] = f'dict(kernel_size={kernel_size}, ' \
                                 f'padding={padding})'
        else:
            head_new['final_layer'] = dict(kernel_size=kernel_size)
            head_append_dict[
                'final_layer'] = f'dict(kernel_size={kernel_size})'
    if 'upsample' in extra:
        neck_new.update(
            dict(
                type='FeatureMapProcessor',
                scale_factor=float(extra['upsample']),
                apply_relu=True,
            ))


def process_has_final_layer(has_final_layer: bool, head_new: Dict,
                            head_deleted_dict: Dict,
                            head_append_dict: Dict) -> None:
    """Process the has_final_layer field and update the head dictionary."""
    head_deleted_dict['has_final_layer'] = str(has_final_layer)
    if not has_final_layer:
        if 'final_layer' not in head_new:
            head_new['final_layer'] = None
        head_append_dict['final_layer'] = 'None'


def check_and_update_config(neck: Optional[ConfigType],
                            head: ConfigType) -> Tuple[Optional[Dict], Dict]:
    """Check and update the configuration of the head and neck components.
    Args:
        neck (Optional[ConfigType]): Configuration for the neck component.
        head (ConfigType): Configuration for the head component.

    Returns:
        Tuple[Optional[Dict], Dict]: Updated configurations for the neck
            and head components.
    """
    head_new, neck_new = head.copy(), neck.copy() if isinstance(neck,
                                                                dict) else {}
    head_deleted_dict, head_append_dict = {}, {}

    if 'input_transform' in head:
        input_transform = head_new.pop('input_transform')
        head_deleted_dict['input_transform'] = f'\'{input_transform}\''
    else:
        input_transform = 'select'

    if 'input_index' in head:
        input_index = head_new.pop('input_index')
        head_deleted_dict['input_index'] = str(input_index)
    else:
        input_index = (-1, )

    if 'align_corners' in head:
        align_corners = head_new.pop('align_corners')
        head_deleted_dict['align_corners'] = str(align_corners)
    else:
        align_corners = False

    process_input_transform(input_transform, head, head_new, head_deleted_dict,
                            head_append_dict, neck_new, input_index,
                            align_corners)

    if 'extra' in head:
        extra = head_new.pop('extra')
        process_extra_field(extra, head_new, head_deleted_dict,
                            head_append_dict, neck_new)

    if 'has_final_layer' in head:
        has_final_layer = head_new.pop('has_final_layer')
        process_has_final_layer(has_final_layer, head_new, head_deleted_dict,
                                head_append_dict)

    display_modifications(head_deleted_dict, head_append_dict, neck_new)

    neck_new = neck_new if len(neck_new) else None
    return neck_new, head_new


@master_only
def display_modifications(head_deleted_dict: Dict, head_append_dict: Dict,
                          neck: Dict) -> None:
    """Display the modifications made to the head and neck configurations.

    Args:
        head_deleted_dict (Dict): Dictionary of deleted fields in the head.
        head_append_dict (Dict): Dictionary of appended fields in the head.
        neck (Dict): Updated neck configuration.
    """
    if len(head_deleted_dict) + len(head_append_dict) == 0:
        return

    old_model_info, new_model_info = build_model_info(head_deleted_dict,
                                                      head_append_dict, neck)

    total_info = '\nThe config you are using is outdated. '\
                 'The following section of the config:\n```\n'
    total_info += old_model_info
    total_info += '```\nshould be updated to\n```\n'
    total_info += new_model_info
    total_info += '```\nFor more information, please refer to '\
                  'https://mmpose.readthedocs.io/en/latest/' \
                  'guide_to_framework.html#step3-model'

    logger: MMLogger = MMLogger.get_current_instance()
    logger.warning(total_info)


def build_model_info(head_deleted_dict: Dict, head_append_dict: Dict,
                     neck: Dict) -> Tuple[str, str]:
    """Build the old and new model information strings.
    Args:
        head_deleted_dict (Dict): Dictionary of deleted fields in the head.
        head_append_dict (Dict): Dictionary of appended fields in the head.
        neck (Dict): Updated neck configuration.

    Returns:
        Tuple[str, str]: Old and new model information strings.
    """
    old_head_info = build_head_info(head_deleted_dict)
    new_head_info = build_head_info(head_append_dict)
    neck_info = build_neck_info(neck)

    old_model_info = 'model=dict(\n' + ' ' * 4 + '...,\n' + old_head_info
    new_model_info = 'model=dict(\n' + ' ' * 4 + '...,\n' \
                     + neck_info + new_head_info

    return old_model_info, new_model_info


def build_head_info(head_dict: Dict) -> str:
    """Build the head information string.

    Args:
        head_dict (Dict): Dictionary of fields in the head configuration.
    Returns:
        str: Head information string.
    """
    head_info = ' ' * 4 + 'head=dict(\n'
    for key, value in head_dict.items():
        head_info += ' ' * 8 + f'{key}={value},\n'
    head_info += ' ' * 8 + '...),\n'
    return head_info


def build_neck_info(neck: Dict) -> str:
    """Build the neck information string.
    Args:
        neck (Dict): Updated neck configuration.

    Returns:
        str: Neck information string.
    """
    if len(neck) > 0:
        neck = neck.copy()
        neck_info = ' ' * 4 + 'neck=dict(\n' + ' ' * 8 + \
                    f'type=\'{neck.pop("type")}\',\n'
        for key, value in neck.items():
            neck_info += ' ' * 8 + f'{key}={str(value)},\n'
        neck_info += ' ' * 4 + '),\n'
    else:
        neck_info = ''
    return neck_info
