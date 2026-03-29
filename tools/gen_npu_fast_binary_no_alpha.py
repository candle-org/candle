#!/usr/bin/env python3
"""Generate thin same-dtype no-alpha NPU fast binary wrappers."""

# (func_name, resolve_op_name, public_name, helper_name)
OPS = [
    ("atan2", "Atan2", "atan2", "_fast_binary_no_alpha_exec"),
    ("pow_tensor_tensor", "PowTensorTensor", "pow", "_fast_binary_two_inputs_exec"),
    ("remainder", "RemainderTensorTensor", "remainder", "_fast_binary_two_inputs_exec"),
    ("fmod", "FmodTensor", "fmod", "_fast_binary_two_inputs_exec"),
    ("logaddexp", "LogAddExp", "logaddexp", "_fast_binary_two_inputs_exec"),
    ("logaddexp2", "LogAddExp2", "logaddexp2", "_fast_binary_two_inputs_exec"),
]


def main() -> None:
    """Print the generated wrapper block to stdout."""
    print("# BEGIN GENERATED SAME-DTYPE NO-ALPHA FAST BINARY OPS")
    for op_name, resolve_name, public_name, helper_name in OPS:
        print(f"# op={op_name} resolve={resolve_name} public={public_name}")
        print(f"cdef object _{op_name}_getws_ptr = None")
        print(f"cdef object _{op_name}_exec_ptr = None")
        print()
        print(f"def fast_{op_name}(a, b):")
        print(f"    return {helper_name}(a, b, _{op_name}_getws_ptr, _{op_name}_exec_ptr, \"{public_name}\")")
        print()
    print("# END GENERATED SAME-DTYPE NO-ALPHA FAST BINARY OPS")


if __name__ == "__main__":
    main()
