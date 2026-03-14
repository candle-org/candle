class TorchDispatchMode:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False
