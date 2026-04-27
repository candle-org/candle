from ._tensor_str import _str, get_printoptions, printoptions, set_printoptions


def format_tensor(tensor, tensor_contents=None):
    if getattr(tensor, "_pending", False):
        from ._dispatch.pipeline import current_pipeline

        pipe = current_pipeline()
        if pipe is not None:
            pipe.flush()

    return _str(tensor, tensor_contents=tensor_contents)
