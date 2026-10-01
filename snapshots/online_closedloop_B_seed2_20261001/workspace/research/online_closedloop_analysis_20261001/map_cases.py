#!/usr/bin/env python3
"""CPU replay of saved actual poses against original PDM drivable polygons.

Only frames 3..10 have saved pre-action rear poses. This is not a full scorer
re-evaluation: NC/TTC and final frame 11 are not reconstructed here.
"""
import sys, pickle
from analyze import *
runtime=read(REG/'runtime.json')
sys.path[:0]=[str(REG.parent/'code/projects/SimEngine'),runtime['nuplan_root']]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from worldengine.components.agents.policy.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap
from worldengine.components.agents.policy.pdm_planner.utils.pdm_array_representation import state_array_to_coords_array

def main():
    cases=read(OUT/'selected_cases.json');manifest={r['scene_id']:r for r in read(REG/'scene_manifest.json')['scenes']};records=[]
    (OUT/'figures').mkdir(exist_ok=True)
    for case in cases:
      scene=case['scene'];row=manifest[scene]
      source=pickle.load(open(row['scene_path'],'rb'))[scene]
      offset=-np.asarray(source['metadata']['old_origin_in_current_coordinate']);vehicle=get_pacifica_parameters()
      map_api=get_maps_api(runtime['map_root'],'nuplan-maps-v1.0',row['map'])
      fig,axs=plt.subplots(1,4,figsize=(17,5));extent=[]
      for ax,(jid,plans) in zip(axs,case['plans'].items()):
        states=np.zeros((1,8,11));states[0,:,:3]=[p['actual_rear_pose'] for p in plans];states[0,:,:2]+=offset
        initial=EgoState.build_from_rear_axle(StateSE2(*states[0,0,:3]),StateVector2D(0,0),StateVector2D(0,0),0,TimePoint(0),vehicle)
        drivable=PDMDrivableMap.from_simulation(map_api,initial,100)
        coords=state_array_to_coords_array(states,vehicle)[0]
        inside=drivable.points_in_polygons(coords.reshape(-1,2)).reshape(len(drivable),8,5)
        offroad=(inside[:,:,:4].sum(0)>0).sum(-1)<4
        first=int(np.flatnonzero(offroad)[0]+3) if offroad.any() else None
        expected=case['timing'][jid]['first_NC_DAC_violation_step'];metrics=case['metrics'][jid]
        # With NC==1, the stored combined violation identifies DAC alone.
        if metrics['no_at_fault_collisions']==1 and metrics['drivable_area_compliance']==0 and expected is not None and expected<=10:
          assert first==expected,(jid,scene,first,expected)
        if metrics['drivable_area_compliance']==1:assert not offroad.any(),(jid,scene)
        track=source['object_track'][source['sdc_id']]['state']
        expert=np.zeros((1,9,11));expert[0,:,:2]=np.asarray(track['position'])[3:12,:2]+offset
        expert[0,:,2]=np.asarray(track['heading'])[3:12]
        expert[0,:,0]-=vehicle.rear_axle_to_center*np.cos(expert[0,:,2])
        expert[0,:,1]-=vehicle.rear_axle_to_center*np.sin(expert[0,:,2])
        expert_coords=state_array_to_coords_array(expert,vehicle)[0]
        expert_inside=drivable.points_in_polygons(expert_coords.reshape(-1,2)).reshape(len(drivable),9,5)
        expert_offroad=(expert_inside[:,:,:4].sum(0)>0).sum(-1)<4
        expert_xy=expert[0,:,:2]-offset
        ax.plot(expert_xy[:,0],expert_xy[:,1],color='#39844a',lw=1,label='logged expert')
        for poly in drivable._geometries:
          polys=list(poly.geoms) if hasattr(poly,'geoms') else [poly]
          for pp in polys:
            xy=np.asarray(pp.exterior.coords)-offset
            ax.add_patch(Polygon(xy,facecolor='#e2e8ee',edgecolor='#acb8c5',lw=.3))
        local=coords-offset
        for i,box in enumerate(local[:,:4]):
          ax.add_patch(Polygon(box,fill=False,edgecolor='#cc3333' if offroad[i] else '#2278a0',lw=1))
          ax.text(local[i,-1,0],local[i,-1,1],str(i+3),fontsize=7)
        xy=states[0,:,:2]-offset;ax.plot(xy[:,0],xy[:,1],color='#2278a0',lw=1)
        # Plans directly before first violation (or step 6 for safe controls).
        target=max(4,min(9,(first or 7)-1));current=next(p for p in plans if p['step']==target);previous=next(p for p in plans if p['step']==target-1)
        old=np.asarray(previous['executed_world_waypoints'])[2:];new=np.asarray(current['executed_world_waypoints'])[1:];count=min(len(old),len(new))
        ax.plot(old[:count,0],old[:count,1],'--',color='#d48c19',label=f'old plan @ {target-1}')
        ax.plot(new[:count,0],new[:count,1],':',color='#8131b8',label=f'new plan @ {target}')
        ax.set_title(jid.replace('dev_train_','').replace('dev_','')+f'\nDAC={metrics["drivable_area_compliance"]:.0f}; first={first}')
        ax.set_aspect('equal');ax.legend(fontsize=7);ax.set_xlabel('scene world x (m)')
        extent.append(local.reshape(-1,2));records.append({'scene':scene,'mode':case['mode'],'category':case['category'],'job':jid,'first_DAC_frame_3_to_10':first,'stored_first_NC_DAC':expected,'DAC':metrics['drivable_area_compliance'],'global_offset':offset.tolist(),'offroad_frames':[int(i+3) for i in np.flatnonzero(offroad)],'logged_expert_offroad_frames':[int(i+3) for i in np.flatnonzero(expert_offroad)],'actual_distance_frames3_10_m':float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()),'mean_speed_frames3_10_mps':float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()/3.5)})
      points=np.concatenate(extent);low=points.min(0)-5;high=points.max(0)+5
      for ax in axs:ax.set_xlim(low[0],high[0]);ax.set_ylim(low[1],high[1])
      fig.suptitle(case['category']+' | '+case['mode']+' | '+scene+'\nSaved actual boxes, frames 3–10; common-time old/new plans. No candidate rewards recorded in evaluator.',fontsize=10)
      fig.tight_layout(rect=[0,0,1,.9]);fig.savefig(OUT/'figures'/f'case_{case["mode"]}_{scene.rsplit("-",1)[1]}.png',dpi=150);plt.close(fig)
      print(case['mode'],scene,'verified',flush=True)
    dump('map_case_audit.json',records)

if __name__=='__main__':main()
