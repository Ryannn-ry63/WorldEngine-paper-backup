"""World-coordinate, common-time plan comparisons and diagnostic one-second reuse."""
import numpy as np


def common_future_difference(previous,current,elapsed_steps=1):
    old=np.asarray(previous,dtype=float)[elapsed_steps+1:]
    new=np.asarray(current,dtype=float)[1:]
    count=min(len(old),len(new))
    if count<1 or not np.isfinite(old[:count]).all() or not np.isfinite(new[:count]).all():
        raise ValueError('No finite overlapping future plan')
    distances=np.linalg.norm(old[:count,:2]-new[:count,:2],axis=1)
    return dict(overlap_samples=count,interval_seconds=.5,
        overlap_mean_distance_m=float(distances.mean()),
        next_step_distance_m=float(distances[0]),max_distance_m=float(distances.max()))


class PlanLatch:
    def __init__(self,interval_steps=1):
        if interval_steps not in (1,2):raise ValueError('Use .5 or 1 second diagnostic interval')
        self.interval_steps=interval_steps;self.previous=None;self.scene=None;self.start=None;self.last_step=None

    def apply(self,trajectory,scene,step):
        from worldengine.common.dataclasses import Trajectory
        fresh=self.scene!=scene or self.last_step is None or step!=self.last_step+1
        if fresh:
            self.previous=None;self.start=step;self.scene=scene
        reused=self.interval_steps==2 and not fresh and (step-self.start)%2==1
        if reused:
            # The environment still advances .5 s. Preserve the previous
            # absolute world plan and its derivatives; never re-anchor it.
            result=Trajectory(**{k:np.asarray(getattr(self.previous,k))[1:].copy() for k in
                ('waypoints','velocities','headings','angular_velocities')})
        else:
            result=trajectory
        delta=common_future_difference(self.previous.waypoints,result.waypoints) if self.previous is not None else None
        self.previous=result;self.last_step=step
        return result,dict(reused=reused,interval_seconds=.5*self.interval_steps,common_future=delta)
