import pytest

from segysak import tools

IN_CROP = [5, 10, 25, 35]
IN_LIMITS = [0, 20, 0, 28]

def test_tools_check_crop():
    assert [5, 10, 25, 28] == tools.check_crop(IN_CROP, IN_LIMITS)

if __name__ == '__main__':
    c = tools.check_crop(IN_CROP, IN_LIMITS)
    print(c) 