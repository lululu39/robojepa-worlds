"""Test vendor namespace isolation without importing Torch on the submit node."""
import ast
from pathlib import Path
import sys
import tempfile
import types
import unittest
import importlib.util


class DecoderImportIsolation(unittest.TestCase):
    def test_upstream_dependencies_do_not_replace_robojepa_modules(self):
        tree=ast.parse(Path('robojepa_worlds/rgb_decoder.py').read_text())
        functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['load_file','upstream_decoder_class']]
        namespace={'sys':sys,'types':types,'importlib':importlib}
        existing={k:v for k,v in sys.modules.items() if k=='src' or k.startswith('src.') or k.startswith('robojepa_worlds._jepawms_')}
        aliases=[]
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);namespace['UPSTREAM']=root
                files={'src/models/utils/modules.py':'class Block: pass\n',
                       'src/models/utils/pos_embs.py':'def get_2d_sincos_pos_embed(): pass\n',
                       'src/utils/tensors.py':'def trunc_normal_(): pass\n',
                       'app/plan_common/models/decoder.py':
                       'from src.models.utils.modules import Block\nfrom src.models.utils.pos_embs import get_2d_sincos_pos_embed\nfrom src.utils.tensors import trunc_normal_\nclass VisionTransformerDecoder(Block): pass\n'}
                for name,content in files.items():
                    p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content)
                original=types.ModuleType('src.models.utils.modules');original.identity='robojepa'
                sys.modules['src.models.utils.modules']=original
                exec(compile(ast.Module(body=functions,type_ignores=[]),'rgb_decoder_imports','exec'),namespace)
                cls=namespace['upstream_decoder_class']()
                self.assertIs(sys.modules['src.models.utils.modules'],original)
                self.assertEqual(cls.__module__,'robojepa_worlds._jepawms_decoder')
                self.assertEqual(cls.__bases__[0].__module__,'robojepa_worlds._jepawms_blocks')
                self.assertIs(namespace['upstream_decoder_class'](),cls)
        finally:
            for key in list(sys.modules):
                if key=='src' or key.startswith('src.') or key.startswith('robojepa_worlds._jepawms_'):
                    del sys.modules[key]
            sys.modules.update(existing)


if __name__=='__main__':unittest.main()
