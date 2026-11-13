"""Historical implementations are preserved; new theme entry points are canonical.

Keep virtual source locations for their relative imports and route defaults to
numbered outputs. Frozen campaign sources and prior results are never edited.
"""
import re
import sys
import types
from pathlib import Path
from .paths import SCRIPTS,output

def load(name,number):
    path=Path(__file__).resolve().parents[1]/'analyses/compat_impl'/name
    module=types.ModuleType('_historical_'+name.replace('.','_'))
    module.__file__=str(SCRIPTS/name);sys.modules[module.__name__]=module
    source=path.read_text()
    own={Path(name).stem, '12_continuous_dynamics'}
    source=re.sub(r'(["\x27]outputs["\x27]\s*/\s*)["\x27]([^"\x27]+)["\x27]',
                  lambda m:m.group(1)+repr(f'{int(number):02d}')+' / "legacy" / '+repr(Path(name).stem) if m.group(2) in own else m.group(0),source)
    # Module references keep their virtual old path, while output defaults change.
    exec(compile(source,str(path),'exec'),module.__dict__)
    return module

def export(namespace,name,number):
    module=load(name,number)
    namespace.update({k:v for k,v in vars(module).items() if not k.startswith('__')})
    if namespace.get('__name__')=='__main__':module.main()
