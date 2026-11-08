"""Nine recorded frames only; requires NumPy, Matplotlib and Pillow. No extrapolation."""
import io,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image
from plot_cases import draw_map,footprint
HERE=Path(__file__).resolve().parent
style=json.loads((HERE/'style.json').read_text())
for n in [30,33]:
 data=json.loads((HERE/'data'/f'case_{n:02d}.json').read_text());models=data['models'];allxy=np.concatenate([m['local_xy'] for m in models.values()]);lo=allxy.min(0)-10;hi=allxy.max(0)+12;lo[1]=min(lo[1],-10);hi[1]=max(hi[1],12);frames=[]
 for i in range(9):
  fig,axs=plt.subplots(1,2,figsize=(10,5))
  for ax,name in zip(axs,['Base','SARA']):
   draw_map(ax,data,style);m=models[name];xy=np.array(m['local_xy']);color=style['colors'][name]
   for p in m['actors'][i]:ax.add_patch(Polygon(p,fc=style['actor_fill'],ec=style['actor_edge'],alpha=.65,lw=.5,zorder=2))
   ax.plot(xy[:i+1,0],xy[:i+1,1],'-o',c=color,lw=2,ms=3,zorder=4);ax.add_patch(Polygon(footprint(xy[i],m['yaw_local'][i]),fc=color,ec=color,alpha=.65,zorder=5))
   ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),xlabel='Longitudinal (m)',ylabel='Lateral (m)',title=f'{name} | t={i*.5:.1f} s | v={m["speed_mps"][i]:.1f} m/s');ax.set_aspect('equal');ax.spines[['top','right']].set_visible(False)
  fig.suptitle(f'{n} | {data["title"]}',fontweight='bold');fig.text(.5,.015,'Recorded poses and actors every 0.5 s; gray = other actors.',ha='center');fig.tight_layout(rect=(0,.09,1,.94));buf=io.BytesIO();fig.savefig(buf,format='png',dpi=110);plt.close(fig);buf.seek(0);frames.append(Image.open(buf).convert('RGB'))
 frames[0].save(HERE.parent/f'case_{n:02d}.gif',save_all=True,append_images=frames[1:],duration=[500]*8+[1500],loop=0);print('Animated',n,flush=True)
