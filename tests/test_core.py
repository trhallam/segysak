from typing import Type
import pytest

from segysak._core import FrozenDict


class TestFrozenDict:
    def test_init(self):
        fdict = FrozenDict({"a": 1})
        assert isinstance(fdict, FrozenDict)

    def test_noset_item(self):
        fdict = FrozenDict({"a": 1})
        with pytest.raises(TypeError):
            fdict["b"] = 2

    def test_del_item(self):
        fdict = FrozenDict({"a": 1})
        with pytest.raises(TypeError):
            del fdict["a"]

    def test_removed_atrs(self):
        fdict = FrozenDict({"a": 1})
        for atr in ("clear", "update", "pop", "popitem", "setdefault"):
            with pytest.raises(AttributeError):
                getattr(fdict, atr)

    def test_atrs(self):
        fdict = FrozenDict({"a": 1})
        for atr in ("keys", "items", "values"):
            try:
                getattr(fdict, atr)
                assert True
            except:
                assert False

    def test_hash(self):
        fdict = FrozenDict({"a": 1})
        assert isinstance(fdict.__hash__(), int)

    def test_repr(self):
        fdict = FrozenDict({"a": 1})
        assert isinstance(str(fdict), str)

    def test_fromkeys(self):
        fdict = FrozenDict().fromkeys(["a"], [1])
        assert isinstance(fdict, FrozenDict)

