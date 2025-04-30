# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

from mmengine.infer import BaseInferencer


def get_model_aliases(scope: str = 'mmpose') -> Dict[str, str]:
    """Retrieve model aliases and their corresponding configuration names.

    Args:
        scope (str, optional): The scope for the model aliases. Defaults
            to 'mmpose'.

    Returns:
        Dict[str, str]: A dictionary containing model aliases as keys and
            their corresponding configuration names as values.
    """

    # Get a list of model configurations from the metafile
    repo_or_mim_dir = BaseInferencer._get_repo_or_mim_dir(scope)
    model_cfgs = BaseInferencer._get_models_from_metafile(repo_or_mim_dir)

    model_alias_dict = dict()
    for model_cfg in model_cfgs:
        if 'Alias' in model_cfg:
            if isinstance(model_cfg['Alias'], str):
                model_alias_dict[model_cfg['Alias']] = model_cfg['Name']
            elif isinstance(model_cfg['Alias'], list):
                for alias in model_cfg['Alias']:
                    model_alias_dict[alias] = model_cfg['Name']
            else:
                raise ValueError(
                    'encounter an unexpected alias type. Please raise an '
                    'issue at https://github.com/open-mmlab/mmpose/issues '
                    'to announce us')

    return model_alias_dict
