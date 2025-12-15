from typing import Callable, Dict, Optional, Type, Any


class Registry:
    """
    Minimal name -> object registry with a decorator helper.
    """

    def __init__(self, kind: str):
        self.kind = kind
        self._map: Dict[str, Any] = {}

    def register(self, name: Optional[str] = None) -> Callable[[Any], Any]:
        def deco(obj: Any) -> Any:
            key = name or getattr(obj, "name", None) or getattr(obj, "__name__", None)
            if not key:
                raise ValueError(f"Cannot register unnamed {self.kind} object: {obj}")
            if key in self._map:
                raise KeyError(f"{self.kind} '{key}' already registered")
            self._map[key] = obj
            return obj
        return deco

    def get(self, name: str) -> Any:
        if name not in self._map:
            raise KeyError(f"Unknown {self.kind}: '{name}'. Available: {sorted(self._map.keys())}")
        return self._map[name]

    def items(self):
        return self._map.items()

    def names(self):
        return list(self._map.keys())


# Global registries for extension points
GRAPH_BUILDERS = Registry("graph_builder")
EMBEDDERS = Registry("embedder")
EXPLAINERS = Registry("explainer")

