""" Functions needed elsewhere in the library.
"""

from attrdict import AttrDict


class FrozenDict(AttrDict):
    """A Frozen Attribute Dictionary for protected key pair mappings in SEGYSAK.
    """

    def __setitem__(self, key, value):
        raise TypeError(
            "%r object does not support item assignment" % type(self).__name__
        )

    def __delitem__(self, key):
        raise TypeError(
            "%r object does not support item deletion" % type(self).__name__
        )

    def __getattribute__(self, attribute):
        if attribute in ("clear", "update", "pop", "popitem", "setdefault"):
            raise AttributeError(
                "%r object has no attribute %r" % (type(self).__name__, attribute)
            )
        return dict.__getattribute__(self, attribute)

    def __hash__(self):
        return hash(tuple(sorted(self.items())))

    def __repr__(self):
        return "FrozenDict({contents})".format(
            contents=super(AttrDict, self).__repr__()
        )

    def fromkeys(self, S, v):
        return type(self)(dict(self).fromkeys(S, v))
