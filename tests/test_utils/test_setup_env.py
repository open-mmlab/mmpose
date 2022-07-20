# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import sys
from unittest import TestCase

from mmengine import DefaultScope

from mmpose.utils import register_all_modules


class TestSetupEnv(TestCase):

    def test_register_all_modules(self):
        from mmpose.registry import DATASETS

        dataset_name = 'CocoDataset'
        dataset_module = 'mmpose.datasets.datasets.body.coco_dataset'

        # not init default scope
        module = dataset_module
        while '.' in module:
            sys.modules.pop(module, None)
            module = module.rsplit('.', 1)[0]
        DATASETS._module_dict.pop(dataset_name, None)
        self.assertFalse(dataset_name in DATASETS.module_dict)
        register_all_modules(init_default_scope=False)
        self.assertTrue(dataset_name in DATASETS.module_dict)

        # init default scope
        module = dataset_module
        while '.' in module:
            sys.modules.pop(module, None)
            module = module.rsplit('.', 1)[0]
        DATASETS._module_dict.pop(dataset_name, None)
        self.assertFalse(dataset_name in DATASETS.module_dict)
        register_all_modules(init_default_scope=True)
        self.assertTrue(dataset_name in DATASETS.module_dict)
        self.assertEqual(DefaultScope.get_current_instance().scope_name,
                         'mmpose')

        # init default scope when another scope is init
        name = f'test-{datetime.datetime.now()}'
        DefaultScope.get_instance(name, scope_name='test')
        with self.assertWarnsRegex(
                Warning, 'The current default scope "test" is not "mmpose"'):
            register_all_modules(init_default_scope=True)
