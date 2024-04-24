""" Functions needed elsewhere in the library.
"""


class MetaDataClass:

    def __iter__(self):
        for match in self.__match_args__:
            yield match

    def keys(self):
        return self.__match_args__

    def values(self):
        for match in self.__match_args__:
            yield self.__getattr__(match)

    def items(self):
        return {key: value for key, value in zip(self.keys(), self.values())}

    def __getitem__(self, item):
        return self.__dict__[item]
