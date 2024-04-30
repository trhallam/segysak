""" Functions needed elsewhere in the library.
"""

from typing import Tuple, Generator, Any, Dict


class MetaDataClass:

    def __iter__(self) -> Generator[str, None, None]:
        return self.keys()

    def keys(self) -> Generator[str, None, None]:
        for field in self.__dataclass_fields__:
            yield field

    def values(self) -> Generator[Any, None, None]:
        for field in self.keys():
            yield self.__getattr__(field)

    def items(self) -> Generator[Tuple[str, Any], None, None]:
        for key in self.keys():
            yield key, self.__getitem__(key)

    def __getitem__(self, item: str) -> Any:
        return self.__dict__[item]
