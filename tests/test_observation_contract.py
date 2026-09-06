"""观测身份、共同可见门禁与同源权重预算回归。"""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import numpy as np
import imageio.v2 as imageio
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lib.multiview_observation import (
    file_identity,write_observation_manifest,selected_map_views,verify_observation_manifest,
    common_observation_gate,source_budget_weights,
)


class ObservationContractTests(unittest.TestCase):
    def test_front_is_never_a_generated_view(self):
        self.assertEqual(selected_map_views(['front']),['front'])
        self.assertEqual(selected_map_views(['back','front','left']),['front','back','left'])
        with self.assertRaises(ValueError): selected_map_views(['front','front'])
        with self.assertRaises(ValueError): selected_map_views(['../back'])

    def test_masks_retain_missing_as_unknown(self):
        seg=np.ones((3,3),bool);strand=np.zeros((3,3,3),np.uint8);strand[...,0]=255
        depth=np.ones((3,3));depth[0,0]=np.nan
        mesh=np.ones((3,3),bool);mesh[1,1]=False
        mask=common_observation_gate(seg,strand,depth,mesh,True)
        self.assertEqual(mask['accepted'].sum(),7)
        self.assertTrue(mask['unknown_or_conflict'][1,1])
        self.assertFalse(common_observation_gate(seg,strand,depth,mesh,False)['accepted'].any())

    def test_duplicate_generated_views_share_budget(self):
        original={'source_kind':'original','source_group':'photo'}
        generated={'source_kind':'generated','source_group':'flux-photo'}
        records={'front':original,'left':generated,'left_copy':generated}
        weights=source_budget_weights({k:np.ones((2,2)) for k in records},records)
        self.assertAlmostEqual(weights['front'].sum(),1)
        self.assertAlmostEqual(weights['left'].sum()+weights['left_copy'].sum(),.5)
        single=source_budget_weights({'front':np.ones((2,2)),'left':np.ones((2,2))},
                                    {'front':original,'left':generated})
        np.testing.assert_allclose(single['left'],weights['left']+weights['left_copy'])

    def test_extra_groups_cannot_evade_generated_cap(self):
        records={str(i):{'source_kind':'generated','source_group':str(i)} for i in range(5)}
        weights=source_budget_weights({k:np.ones(4) for k in records},records)
        self.assertAlmostEqual(sum(x.sum() for x in weights.values()),.5)
        with self.assertRaises(ValueError):
            source_budget_weights({'x':np.array([np.nan])},{'x':records['0']})

    def test_hashes_and_partial_manifest_are_rejected(self):
        output=Path(__file__).parent/'outputs/volume_partition_integration';output.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output) as temp:
            path=Path(temp)/'asset';path.write_bytes(b'first')
            asset=file_identity(path)
            record={'source_kind':'original','source_group':'photo','image':asset,
                    'products':{k:asset for k in ('strand_map','depth_map','seg')}}
            manifest={'version':1,'status':'complete','observations':{'front':record}}
            self.assertIn('front',verify_observation_manifest(manifest,['front']))
            manifest['status']='building'
            with self.assertRaises(ValueError):verify_observation_manifest(manifest,['front'])
            manifest['status']='complete';path.write_bytes(b'changed')
            with self.assertRaises(ValueError):verify_observation_manifest(manifest,['front'])

    def test_front_only_producer_preserves_source_and_has_one_stack_entry(self):
        import scripts.render.compute_multiview_maps as producer
        output=Path(__file__).parent/'outputs/volume_partition_integration';output.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output) as temp:
            directory=Path(temp);rgb=np.full((512,512,3),255,np.uint8)
            source=directory/'front.png';imageio.imwrite(source,rgb)
            depth=directory/'front.npy';np.save(depth,np.ones((512,512),np.float32))
            checkpoint=directory/'model.pth';checkpoint.write_bytes(b'mocked-existing-checkpoint')
            out=directory/'maps';(out/'seg').mkdir(parents=True);(out/'body_img').mkdir()
            imageio.imwrite(out/'seg/front.png',rgb);imageio.imwrite(out/'body_img/front.png',rgb)
            argv=['compute_multiview_maps','--front_img',str(source),'--front_strand',str(source),
                  '--front_depth',str(depth),'--views','front','--out_dir',str(out),
                  '--checkpoint_img2strand',str(checkpoint),'--checkpoint_img2depth',str(checkpoint)]
            model=Mock();model.to.return_value=model
            with patch.object(sys,'argv',argv),patch.object(producer,'run_sam3_masks'),\
                 patch.object(producer,'stage_resized'),patch.object(producer,'BaseOptions') as options,\
                 patch.object(producer,'create_img2strand_model',return_value=model),\
                 patch.object(producer,'DepthModel',return_value=model),\
                 patch.object(producer.torch.nn,'DataParallel',return_value=model),\
                 patch.object(producer.torch,'load',return_value={}),\
                 patch.object(producer,'predict_strand',side_effect=AssertionError('front overwrite')),\
                 patch.object(producer,'depth2vis'):
                producer.main()
            manifest=json.loads((out/'observation_manifest.json').read_text())
            self.assertEqual(manifest['requested_views'],['front'])
            verify_observation_manifest(manifest,['front'])
            self.assertEqual(np.load(out/'combined_strand.npz')['views'].tolist(),['front'])
            np.testing.assert_array_equal(imageio.imread(source),rgb)


if __name__=='__main__':
    unittest.main()
