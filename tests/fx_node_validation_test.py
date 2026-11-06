"""Typed FX nodes reject malformed graphs before any device is initialized."""

import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import gaudirt


def main():
    node, op, program = gaudirt._C.FxNode, gaudirt._C.FxOp, gaudirt._C.BucketProgram
    program([1, 0], [node.linear(0, 1, None, True)], [2])

    def rejects(roles, nodes, outputs, exception=RuntimeError):
        try:
            program(roles, nodes, outputs)
        except exception:
            return
        raise AssertionError("malformed FX program was accepted")

    rejects([1], [[1, 0, 0, -1]], [1], TypeError)
    rejects([1, 1], [node.binary(op.add, 0, 2)], [2])
    rejects([1], [node.unary(op.add, 0)], [1])
    rejects([1], [node.scalar(op.scalar_mul, 0, float("nan"), False)], [1])
    rejects([1], [node.slice(0, -1, 4, 8)], [1])
    rejects([1, 1, 0], [node.fused_norm(0, 1, 2, 1e-6),
                         node.unary(op.fused_residual, 3), node.unary(op.fused_residual, 3)], [4, 5])
    assert not gaudirt.is_initialized()
    print("typed FX validation rejected malformed nodes and legacy integer arrays", flush=True)


if __name__ == "__main__":
    main()
