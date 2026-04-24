"""Minimal torch.overrides stub for candle."""
import functools

def has_torch_function(args):
    return False

def handle_torch_function(func, args, *rest, **kwargs):
    return func(*rest, **kwargs)

def has_torch_function_unary(arg):
    return False

def has_torch_function_variadic(*args):
    return False

def get_default_nowrap_functions():
    return set()

def _is_torch_function_mode_enabled():
    return False
