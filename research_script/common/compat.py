"""Load only the numeric implementations required by maintained research themes."""
import importlib.util
import sys
from pathlib import Path


def load(name, number=None):
    path = Path(__file__).resolve().parents[1]/'analyses/compat_impl'/name
    if path.parent.name != 'compat_impl' or not path.is_file():
        raise ImportError(f'No retained research implementation: {name}')
    module_name = '_research_impl_'+path.stem
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def export(namespace, name, number):
    module = load(name, number)
    namespace.update({k:v for k,v in vars(module).items() if not k.startswith('__')})
    if namespace.get('__name__') == '__main__':
        module.main()
