"""Minimal torch.utils.hooks stub for candle compatibility."""


class RemovableHandle:
    """A handle which provides the capability to remove a hook."""
    __slots__ = ('id', 'hooks_dict')

    def __init__(self, hooks_dict):
        self.id = id(self)
        self.hooks_dict = hooks_dict

    def remove(self):
        if self.id in self.hooks_dict:
            del self.hooks_dict[self.id]

    def __getstate__(self):
        return (self.id, self.hooks_dict)

    def __setstate__(self, state):
        self.id, self.hooks_dict = state
