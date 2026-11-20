import importlib
import pkgutil
import exercises
from kernellab.demo_copy import SPEC as DEMO


def discover():
    specs = {DEMO.name: DEMO}
    for module in pkgutil.iter_modules(exercises.__path__):
        if module.name.startswith("_"):
            continue
        loaded = importlib.import_module(f"exercises.{module.name}")
        spec = getattr(loaded, "SPEC", None)
        if spec is not None:
            if spec.name in specs:
                raise ValueError(f"Duplicate exercise: {spec.name}")
            specs[spec.name] = spec
    return specs
