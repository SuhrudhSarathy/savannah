# src/savannah/__init__.py
import sys

# Python 3.12 compatibility shim for legacy robotics libraries
try:
    import imp
except ImportError:
    import importlib.util
    from types import ModuleType

    imp = ModuleType("imp")
    sys.modules["imp"] = imp
