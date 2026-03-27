"""candle.utils.model_zoo — compatibility stub for torch.utils.model_zoo.

Some third-party libraries (for example older torchvision versions) import::

    from torch.utils.model_zoo import load_url

This module provides that symbol as an alias for
:func:`candle.hub.load_state_dict_from_url`.
"""

from candle.hub import load_state_dict_from_url as load_url

__all__ = ["load_url"]
