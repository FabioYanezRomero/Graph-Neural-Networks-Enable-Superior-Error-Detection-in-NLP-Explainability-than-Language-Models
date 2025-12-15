"""Graph builders: constituency, syntactic, and more.

To add a new builder, create a module in this package and register your
BaseTreeGenerator subclass with the `GENERATORS` registry:

    from .base_generator import BaseTreeGenerator
    from .registry import GENERATORS

    @GENERATORS.register("my_new_tree")
    class MyNewTree(BaseTreeGenerator):
        ...

It will be auto-discovered by `tree_generator.py` at runtime.
"""

from .registry import GENERATORS  # re-export for convenience
