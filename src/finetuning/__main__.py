"""Entrypoint for finetuning package."""
import sys
from .finetuner import main

if __name__ == "__main__":
    sys.exit(main())
