#!/usr/bin/env python3

import sys

sys.path.insert(0, ".")

from kernels import describe_stages


if __name__ == "__main__":
    for stage in describe_stages():
        print(stage)
