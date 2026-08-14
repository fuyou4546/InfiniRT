#!/usr/bin/env python3
import argparse
import os
import sys

import triton
from triton.compiler import ASTSource


@triton.jit
def empty_kernel():
    pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    compiled = triton.compile(
        ASTSource(fn=empty_kernel, signature={}, constexprs={}))
    binary = next(v for v in compiled.asm.values()
                  if isinstance(v, (bytes, bytearray)))

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "wb") as f:
        f.write(binary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
