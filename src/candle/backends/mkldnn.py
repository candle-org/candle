class _MKLDNN:
    @staticmethod
    def is_available():
        return False


is_available = _MKLDNN.is_available
