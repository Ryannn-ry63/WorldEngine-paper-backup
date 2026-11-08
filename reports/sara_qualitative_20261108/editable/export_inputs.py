"""Server-only exporter; the resulting data/ and plot_cases.py are portable."""
import json, sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import analyze as a

def main():
    results=a.read_results()
    for folder in [HERE.parent,HERE.parent/'round2',HERE.parent/'round3']:
        selected=json.loads((folder/'selected_cases.json').read_text())
        for r in selected['cases']:
            frames=a.extract_scene(results,r['scene']); layers=a.vector_map(frames)
            origin=frames['Base'][3]['ego2global_translation'][:2]; heading=a.yaw(frames['Base'][3])
            trace=json.loads((folder/r['artifacts']['trajectory']).read_text())
            data={'number':r['number'],'scene':r['scene'],'title':r['title_en'],'mode':'R','metrics':r['metrics'],
                  'provenance':{'metadata_sha256':{k:v['sha256'] for k,v in r['metadata'].items()},
                  'base_model_sha256':selected['base_model_sha256'],'sara_model_sha256':selected['sara_model_sha256']},
                  'map':[],'models':{},'coordinate_frame':'Meters relative to initial rear axle; x along initial heading, y left.'}
            for layer,gdf in layers.items():
                for geom in gdf.geometry:
                    if geom is None: continue
                    parts=list(geom.geoms) if geom.geom_type.startswith('Multi') else [geom]
                    for g in parts:
                        if g.geom_type not in ('Polygon','LineString'): continue
                        coords=g.exterior.coords if g.geom_type=='Polygon' else g.coords
                        data['map'].append({'layer':layer,'type':g.geom_type,'xy':a.transform(coords,origin,heading).tolist()})
            for name,fs in frames.items():
                tr=trace['models'][name]
                np.testing.assert_allclose(tr['global_xy'],[f['ego2global_translation'][:2] for f in fs[3:]],rtol=0,atol=1e-9)
                xy=np.array(tr['local_xy']); models={**tr,'yaw_local':[float(a.yaw(f)-heading) for f in fs[3:]],'actors':[]}
                models['time_s']=[(t-tr['timestamp_us'][0])/1e6 for t in tr['timestamp_us']]
                for i,f in enumerate(fs[3:]):
                    ang=a.yaw(f)-heading; c,s=np.cos(ang),np.sin(ang); actors=[]
                    for box in f['anns']['gt_boxes']:
                        center=box[:2]@np.array([[c,s],[-s,c]])+xy[i]
                        actors.append(a.footprint(center,ang+box[6],box[3],box[4],False).tolist())
                    models['actors'].append(actors)
                data['models'][name]=models
            (HERE/'data'/f"case_{r['number']:02d}.json").write_text(json.dumps(data,separators=(',',':')))
            print('exported',r['number'],flush=True)

if __name__=='__main__': main()
