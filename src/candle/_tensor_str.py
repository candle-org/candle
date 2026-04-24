"""Minimal torch._tensor_str stub for candle compatibility."""
import numpy as np


def _str(self, tensor_contents=None):
    """Return a string representation of the tensor."""
    if self.numel() == 0:
        return "tensor([])"

    # Build the header
    header = "tensor("

    # Get numpy representation
    arr = self.numpy()

    # Format the array
    with np.printoptions(threshold=1000, linewidth=80, precision=4):
        body = np.array2string(arr, separator=", ")

    # Indent multi-line arrays
    if "\n" in body:
        body = body.replace("\n", "\n" + " " * 7)
        return header + "\n" + " " * 7 + body + ")"

    return header + body + ")"
