"""Adaptive, label-aware RK4 stages for the strict volume path."""
import torch


def adaptive_partition_rk4(points, reference, query, max_step, min_step,
                           max_angle_degrees=20.0, max_retries=8):
    """Retry low-support or high-curvature stages with half the local step.

    query returns a direction and a low-support mask. Direction sign is made
    continuous within each trial; all stages must be supported before accepting.
    The caller still checks the final corrected segment against the volume.
    """
    if not 0 < min_step <= max_step or max_retries < 0:
        raise ValueError("Invalid adaptive RK4 step budget")
    count = points.shape[1]
    step = torch.full((count,), float(max_step),device=points.device,dtype=points.dtype)
    retries = torch.zeros(count,device=points.device,dtype=torch.long)
    accepted = torch.zeros(count,device=points.device,dtype=torch.bool)
    output = torch.zeros_like(points)

    def align(direction, target):
        if target is None:
            return direction
        return torch.where((direction*target).sum(0,keepdim=True)<0,-direction,direction)

    cosine_limit = torch.cos(torch.tensor(max_angle_degrees*torch.pi/180,device=points.device))
    for attempt in range(max_retries+1):
        k1,l1=query(points)
        k1=align(k1,reference)
        k2,l2=query(points+.5*step*k1);k2=align(k2,k1)
        k3,l3=query(points+.5*step*k2);k3=align(k3,k2)
        k4,l4=query(points+step*k3);k4=align(k4,k3)
        curvature=torch.minimum((k1*k2).sum(0),torch.minimum((k2*k3).sum(0),(k3*k4).sum(0)))
        good=~(l1|l2|l3|l4)&(curvature>=cosine_limit)
        newly=good&~accepted
        output[:,newly]=((k1+2*k2+2*k3+k4)/6)[:,newly]
        accepted|=good
        retry=~accepted&(step>min_step)&(attempt<max_retries)
        if not retry.any():
            break
        step[retry]=(step[retry]*.5).clamp_min(min_step)
        retries[retry]+=1
    output[:,~accepted]=0
    return output,step,~accepted,retries
