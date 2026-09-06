"""分区插值与受限转向不能跨越标签或制造跳跃。"""
import numpy as np
from scripts.recon_3d.recover_boundary_strands import sample_partition_field, cone_directions, grow
from tests.test_head_guard_comparison import PlaneSurface


def test_interpolation_never_averages_opposite_partition():
    lab=np.ones((4,3,3),int);lab[2:]=2
    f=np.zeros((3,4,3,3));f[0,:2]=1;f[0,2:]=-1
    p=np.array([[1.49,1,1],[1.51,1,1]])
    v,ok=sample_partition_field(p,np.array([1,2]),f,lab,np.zeros(3),np.array([3,2,2]))
    assert ok.all()
    np.testing.assert_allclose(v,[[1,0,0],[-1,0,0]])


def test_cone_is_bounded_and_unit_length():
    ref=np.array([[0.,0.,1.],[1.,0.,0.]])
    cone=cone_directions(ref)
    np.testing.assert_allclose(np.linalg.norm(cone,axis=2),1,atol=1e-7)
    assert (np.sum(cone*ref[:,None],axis=2)>=.5-1e-7).all()


def test_blocked_growth_stays_inside_partition_and_preserves_root():
    labels=np.ones((9,9,9),int);labels[5:]=2
    field=np.zeros((3,9,9,9));field[0]=1;field[1]=1
    roots=np.array([[.049, .02, .04]])
    strands,reasons,steps,_,_,_=grow(field,labels,roots,np.array([1]),np.zeros(3),np.full(3,.08),{'solid_mask':labels==0,'conflict':labels==0},PlaneSurface(),max_steps=20,budget=.02)
    np.testing.assert_allclose(strands[:,0],roots,atol=1e-8)
    # 根点原本已在第二分区，不允许通过插值偷偷进入该区。
    assert steps[0]==0


def test_legal_root_can_turn_along_wall_without_crossing():
    from lib.recon_strategy.volume_segment_guard import audit_volume_strands
    labels=np.ones((9,9,9),int);labels[5:]=2
    field=np.zeros((3,9,9,9));field[0]=1;field[1]=1
    roots=np.array([[.041, .02, .04]])
    lo=np.zeros(3);hi=np.full(3,.08)
    strands,reasons,steps,recovered,_,_=grow(field,labels,roots,np.array([1]),lo,hi,{'solid_mask':labels==0,'conflict':labels==0},PlaneSurface(),max_steps=30,budget=.02)
    assert recovered[0]>0
    assert np.linalg.norm(np.diff(strands,axis=1),axis=2).sum()>.015
    assert strands[...,0].max()<.045
    assert audit_volume_strands(strands,np.array([1]),labels,lo,hi)['passed']
