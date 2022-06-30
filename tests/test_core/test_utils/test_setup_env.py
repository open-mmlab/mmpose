# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import sys
from unittest import TestCase

from mmengine import DefaultScope

from mmpose.utils import register_all_modules


class TestSetupEnv(TestCase):

    def test_register_all_modules(self):
        from mmpose.registry import DATASETS

        # not init default scope
        sys.modules.pop('mmpose.datasets', None)
        for key in tuple(sys.modules.keys()):
            if key.startswith('mmpose.datasets.datasets2'):
                sys.modules.pop(key)
        DATASETS._module_dict.pop('CocoDataset', None)
        self.assertFalse('CocoDataset' in DATASETS.module_dict)
        register_all_modules(init_default_scope=False)
        self.assertTrue('CocoDataset' in DATASETS.module_dict)

        # init default scope
        sys.modules.pop('mmpose.datasets')
        for key in tuple(sys.modules.keys()):
            if key.startswith('mmpose.datasets.datasets2'):
                sys.modules.pop(key)
        DATASETS._module_dict.pop('CocoDataset', None)
        self.assertFalse('CocoDataset' in DATASETS.module_dict)
        register_all_modules(init_default_scope=True)
        self.assertTrue('CocoDataset' in DATASETS.module_dict)
        self.assertEqual(DefaultScope.get_current_instance().scope_name,
                         'mmpose')

        # init default scope when another scope is init
        name = f'test-{datetime.datetime.now()}'
        DefaultScope.get_instance(name, scope_name='test')
        with self.assertWarnsRegex(
                Warning, 'The current default scope "test" is not "mmpose"'):
            register_all_modules(init_default_scope=True)
