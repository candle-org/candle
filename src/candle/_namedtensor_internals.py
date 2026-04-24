"""Minimal torch._namedtensor_internals stub for candle."""
def check_serializing_named_tensor(tensor):
    pass
def is_ellipsis(item):
    return item is Ellipsis
def resolve_ellipsis(indices, shape):
    return list(indices)
def single_ellipsis_index(indices, shape):
    return Ellipsis
def unzip_namedshape(namedshape):
    return [], []
def update_names(tensor, names, inplace=False):
    return tensor
