"""Shared progress bar class for SEGY-SAK"""

from typing import Any, Dict, Union
import sys
import warnings
import inspect

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from tqdm import tqdm
    from tqdm.notebook import tqdm_notebook
    from tqdm.autonotebook import tqdm as tqdm_auto

try:
    from IPython import display
except ImportError:
    pass


class Progress:
    """Progress class tracks the shared state of all progress trackers in SEGY-SAK"""

    _segysak_tqdm_func = tqdm_auto
    _segysak_tqdm_kwargs = {
        "disable": False,
        "leave": False,
        "unit_scale": True,
    }

    @classmethod
    def silent(cls):
        return cls._segysak_tqdm_kwargs["disable"]

    def __init__(self, tqdm_func: Union[str, None] = None, **tqdm_kwargs):
        if tqdm_func == "a":
            self.tqdm_func = tqdm_auto
        elif tqdm_func == "t":
            self.tqdm_func = tqdm
        elif tqdm_func == "n":
            self.tqdm_func = tqdm_notebook
        else:
            self.tqdm_func = self._segysak_tqdm_func

        self.tqdm_kwargs = tqdm_kwargs

    def __enter__(self):
        kwargs = self._segysak_tqdm_kwargs.copy()
        kwargs.update(self.tqdm_kwargs)
        self.pbar = self.tqdm_func(**kwargs)
        return self.pbar

    def __exit__(self, type: Any, value: Any, traceback: Any):
        self.pbar.clear()
        self.pbar.close()

    @classmethod
    def set_defaults(cls, **tqdm_kwargs: Dict[str, Any]) -> None:
        """Set the default arguments for tqdm progress reporting in SEGY-SAK.

        The defaults are Global and affect all tqdm reporting in the active Python
        session.

        Args:
            tqdm_kwargs: Any valid [`tqdm.tqdm` argument](https://tqdm.github.io/docs/tqdm/).
        """
        valid_args = inspect.signature(tqdm.__init__).parameters
        for kwarg in tqdm_kwargs:
            try:
                assert kwarg in valid_args
            except AssertionError:
                raise AssertionError(
                    f"{kwarg} is not a valid tqdm argument -> {valid_args.keys()}"
                )
        cls._segysak_tqdm_kwargs.update(tqdm_kwargs)
