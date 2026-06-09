_EXTERNAL_NAMESPACES = set()


def is_external_namespace(name):
    return name in _EXTERNAL_NAMESPACES


class _OpSchema:
    def __init__(self, name, overload_name=""):
        self.name = name
        self.overload_name = overload_name


class _Op:
    def __init__(self, return_value=None, qualname=None, overload_name=""):
        self._return_value = return_value
        self._qualname = qualname
        self._schema = _OpSchema(qualname, overload_name) if qualname else None

    def __call__(self, *args, **kwargs):
        return self._return_value

    @property
    def default(self):
        if self._qualname is None:
            return self
        return _Op(
            self._return_value,
            qualname=self._qualname,
            overload_name="default",
        )


class _Namespace:
    def __init__(self, name):
        self._name = name
        _EXTERNAL_NAMESPACES.add(name)

    def __getattr__(self, name):
        return _Op(None, qualname=f"{self._name}::{name}")


class _TorchVisionNamespace(_Namespace):
    def __init__(self):
        super().__init__("torchvision")

    def __getattr__(self, name):
        if name == "_cuda_version":
            return _Op(0, qualname="torchvision::_cuda_version")
        return super().__getattr__(name)


class _Ops:
    def __init__(self):
        self.torchvision = _TorchVisionNamespace()

    def load_library(self, *_args, **_kwargs):
        return None

    def __getattr__(self, name):
        return _Namespace(name)


ops = _Ops()
