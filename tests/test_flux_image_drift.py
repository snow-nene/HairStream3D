import cv2
import numpy as np
from scripts.vis.audit_flux_image_drift import audit


def test_independent_profile_measurement_recovers_known_translation():
    source=np.zeros((512,512,3),np.uint8)
    cv2.rectangle(source,(110,120),(350,440),(150,150,150),-1)
    target=cv2.warpAffine(source,np.array([[1.,0.,3.],[0.,1.,0.]]),(512,512))
    report,_,_,_=audit(source,target,np.zeros((512,512),bool),'left')
    for value in report['profile_band_threshold_sweep'].values():
        assert value['rows']==118
        assert value['absolute_dx_px_q50_q90']==[3.,3.]
        assert value['signed_dx_median_px']==3.
