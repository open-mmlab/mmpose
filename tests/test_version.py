import mmcv

import mmpose


def test_version():
    version = mmpose.__version__
    mmcv_version = mmpose.digit_version(mmcv.__version__)
    assert isinstance(version, str)
    assert isinstance(mmpose.short_version, str)
    assert mmpose.short_version in version
    assert mmcv_version == mmpose.mmcv_version
