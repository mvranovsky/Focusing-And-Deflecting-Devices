#!/bin/bash


if [ "$#" -ne 0 ]; then
    echo "Usage: $0 <TString Argument>"
    exit 1
fi

python3 parallelFocused.py > output.out 2>output.err &