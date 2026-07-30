"""Late-bound access to CLI composition dependencies.

Command modules use this proxy at call time so compatibility monkeypatches on
``apps.runner.main`` continue to affect the extracted implementations without
creating an import cycle during command registration.
"""

import sys
from typing import Any


class _CompositionProxy:
    def __getattr__(self, name: str) -> Any:
        main = sys.modules.get("apps.runner.main")
        if main is None:
            candidate = sys.modules.get("__main__")
            spec = getattr(candidate, "__spec__", None)
            if getattr(spec, "name", None) == "apps.runner.main":
                main = candidate
        if main is None:
            raise AttributeError(name)
        return getattr(main, name)


composition: Any = _CompositionProxy()
