"""生成入口与图谱来源绑定的无 GPU 接口回归；推理用替身。"""
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lib.multiview_observation import file_identity,generation_source


class GenerationProvenanceTests(unittest.TestCase):
    def test_generation_order_preserves_seeds_and_front_identity(self):
        output=Path(__file__).parent/'outputs/volume_partition_integration';output.mkdir(parents=True,exist_ok=True)
        entry=Path(__file__).resolve().parents[1]/'scripts/infer_2d/flux_redraw_multiview.py'
        class FakePipeline:
            @classmethod
            def from_pretrained(cls,*args,**kwargs):return cls()
            def to(self,device):return self
            def __call__(self,**kwargs):return SimpleNamespace(images=[kwargs['image']])
        with tempfile.TemporaryDirectory(dir=output) as temp:
            root=Path(temp).resolve();data=root/'results/multiview_data/test_image'
            (data/'blender_renders').mkdir(parents=True)
            image=Image.new('RGB',(16,16),'white');image.save(data/'raw_img.png')
            for v in ['front','back']:image.save(data/'blender_renders'/f'{v}.png')
            model=root/'model';model.mkdir();(model/'model_index.json').write_text('{}')
            original=file_identity(data/'raw_img.png')
            with patch('os.execv'),patch.dict(sys.modules,{'diffusers':SimpleNamespace(Flux2KleinInpaintPipeline=FakePipeline)}):
                module=runpy.run_path(str(entry),run_name='test_flux')
            cwd=Path.cwd()
            try:
                os.chdir(root)
                seeds=[]
                for views in [['front','back'],['back','front']]:
                    argv=['flux','--img_id','test_image','--views',*views,'--model',str(model),'--size','16']
                    with patch.object(sys,'argv',argv):module['main']()
                    m=json.loads((data/'flux_redrawn/generation_manifest.json').read_text())
                    seeds.append({k:v['seed'] for k,v in m['observations'].items()})
                    self.assertEqual(m['observations']['front']['source_kind'],'generated')
                self.assertEqual(seeds[0],seeds[1])
                self.assertEqual(original,file_identity(data/'raw_img.png'))
                source=generation_source(data/'flux_redrawn','back',data/'flux_redrawn/back.png',original)
                self.assertEqual(source['status'],'verified')
                (data/'flux_redrawn/back.png').write_bytes(b'changed')
                with self.assertRaises(ValueError):generation_source(data/'flux_redrawn','back',data/'flux_redrawn/back.png',original)
            finally:os.chdir(cwd)

    def test_legacy_history_is_not_invented(self):
        output=Path(__file__).parent/'outputs/volume_partition_integration'
        with tempfile.TemporaryDirectory(dir=output) as temp:
            result=generation_source(temp,'left',Path(temp)/'missing.png',{'sha256':'photo'})
            self.assertEqual(result['status'],'unverified_legacy')


if __name__=='__main__':
    unittest.main()
