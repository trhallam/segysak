""" Functions needed elsewhere in the library.
"""

# from attrdict import AttrDict
from addict import Dict


class FrozenDict(Dict):
    """A Frozen Attribute Dictionary for protected key pair mappings in SEGY-SAK."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.freeze()

    def _is_frozen(self):
        return hasattr(self, "__frozen") and object.__getattribute__(self, "__frozen")

    def __setitem__(self, key, value):
        if self._is_frozen():
            raise TypeError(
                "%r object does not support item assignment" % type(self).__name__
            )
        else:
            super().__setitem__(key, value)

    def __delitem__(self, key):
        if self._is_frozen():
            raise TypeError(
                "%r object does not support item deletion" % type(self).__name__
            )
        else:
            super.__delitem__(key)

    # def __getattribute__(self, attribute):
    #     if attribute in ("clear", "update", "pop", "popitem", "setdefault"):
    #         raise AttributeError(
    #             "%r object has no attribute %r" % (type(self).__name__, attribute)
    #         )
    #     return dict.__getattribute__(self, attribute)

    def __hash__(self):
        return hash(tuple(sorted(self.items())))

    def __repr__(self):
        return "FrozenDict({contents})".format(contents=super(Dict, self).__repr__())

    def fromkeys(self, S, v):
        return type(self)(dict(self).fromkeys(S, v))
