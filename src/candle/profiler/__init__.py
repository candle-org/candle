from .common import ProfilerActivity, ProfilerAction
from .profiler import profile, record_function, schedule


class _ITT:
    @staticmethod
    def is_available():
        return False


itt = _ITT()

__all__ = ["profile", "record_function", "schedule", "ProfilerActivity", "ProfilerAction", "itt"]
