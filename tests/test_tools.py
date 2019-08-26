import pytest

from segysak import tools

IN_CROP = [5, 10, 25, 35]
IN_LIMITS = [0, 20, 0, 28]

if __name__ == '__main__':
    c = tools.check_crop(IN_CROP, IN_LIMITS)
    print(c)