"""可独立运行：python plot_cases.py；依赖 numpy matplotlib pillow。
改 STYLE 修改字体/颜色/视窗；改 TIMES 修改截图时刻。数据保留 0..4 秒全部真实记录。
"""
import argparse,json,io
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image
HERE=Path(__file__).resolve().parent
STYLE={'colors':{'Base':'#cc4c3b','SARA':'#197bbd'},'font_size':11,'font_family':'DejaVu Sans','dpi':200,'figsize':[12,8], 'road_fill':'#eef0f2','map_edge':'#c2c7cc','actor_fill':'#89939c','actor_edge':'#59636e','trajectory_linewidth':2,'marker_size':3,'show_start_body':True,'show_title':True,'case_overrides':{}}
TIMES=[2.,4.]

def footprint(xy,theta):
 p=np.array([[2.588,1.1485],[2.588,-1.1485],[-2.588,-1.1485],[-2.588,1.1485]]);p[:,0]+=1.461;c,s=np.cos(theta),np.sin(theta)
 return p@np.array([[c,s],[-s,c]])+xy

def limits(data):
 xy=np.concatenate([m['local_xy'] for m in data['models'].values()]);lo=xy.min(0)-8;hi=xy.max(0)+10;lo[1]=min(lo[1],-9);hi[1]=max(hi[1],9)
 custom=STYLE['case_overrides'].get(str(data.get('number','')),{})
 return custom.get('xlim',[lo[0],hi[0]]),custom.get('ylim',[lo[1],hi[1]])

def panel(ax,data,name,i):
 xlim,ylim=limits(data)
 for shape in data['map']:
  p=np.array(shape['xy'])
  if shape['type']=='Polygon':ax.add_patch(Polygon(p,fc=STYLE['road_fill'] if shape['layer']!='crosswalks' else '#dce0e3',ec=STYLE['map_edge'],lw=.5,zorder=0))
  else:ax.plot(p[:,0],p[:,1],c=STYLE['map_edge'],lw=.6,ls='--' if shape['layer']=='baseline_paths' else '-',zorder=1)
 for shape in data['drivable_context']:
  for ring in [shape['outer'],*shape['holes']]:
   pts=np.array(ring);ax.plot(pts[:,0],pts[:,1],c='#646d73',lw=.9,zorder=2)
 m=data['models'][name];xy=np.array(m['local_xy']);color=STYLE['colors'][name]
 for p in m['actors'][i]:
  p=np.array(p);c=p.mean(0)
  if xlim[0]-6<c[0]<xlim[1]+6 and ylim[0]-6<c[1]<ylim[1]+6:ax.add_patch(Polygon(p,fc=STYLE['actor_fill'],ec=STYLE['actor_edge'],alpha=.8,lw=.6,zorder=3))
 if STYLE['show_start_body']:ax.add_patch(Polygon(footprint(xy[0],m['yaw_local'][0]),fc='none',ec=color,ls=':',lw=1,zorder=4))
 ax.plot(xy[:i+1,0],xy[:i+1,1],'-o',c=color,lw=STYLE['trajectory_linewidth'],ms=STYLE['marker_size'],zorder=5)
 ax.add_patch(Polygon(footprint(xy[i],m['yaw_local'][i]),fc=color,ec='#222222',alpha=.82,lw=.8,zorder=6))
 for shape in m['outside_polygons'][i]:ax.add_patch(Polygon(shape,fc='#e22e91',ec='#99165e',hatch='///',alpha=.95,lw=.5,zorder=8))
 ax.scatter(*xy[0],s=16,c='black',marker='s',zorder=7)
 ax.set(xlim=xlim,ylim=ylim,xlabel='Initial-heading longitudinal axis (m)',ylabel='Lateral axis (m)',title=f'{name} | t = {m["time_s"][i]:g} s | v = {m["speed_mps"][i]:.1f} m/s');ax.set_aspect('equal');ax.spines[['top','right']].set_visible(False)

def snapshots(data,out,formats):
 fig,axs=plt.subplots(2,len(TIMES),figsize=STYLE['figsize'],squeeze=False)
 for row,name in enumerate(['Base','SARA']):
  for col,t in enumerate(TIMES):
   times=np.array(data['models'][name]['time_s']);assert np.any(np.isclose(times,t));i=int(np.argmin(abs(times-t)));panel(axs[row,col],data,name,i)
 if STYLE['show_title']:fig.suptitle(f'{data.get("display_id","")} | {data.get("title",data["scene"].split("-")[-1])}',fontweight='bold')
 fig.text(.5,.025,'Fixed initial coordinates. Dark outline: auxiliary road union; magenta: ego area outside union. Dotted: initial pose.',ha='center',fontsize=9)
 fig.subplots_adjust(left=.07,right=.97,bottom=.1,top=.91,hspace=.32,wspace=.2)
 for fmt in formats:
  path=out/f'case_{data["number"]:02d}_2s_4s.{fmt}';fig.savefig(path,dpi=STYLE['dpi'])
  if fmt=='svg':path.write_text('\n'.join(l.rstrip() for l in path.read_text().splitlines())+'\n')
 plt.close(fig)

def animate(data,out):
 frames=[]
 for i in range(9):
  fig,axs=plt.subplots(1,2,figsize=(12,4.6))
  for ax,name in zip(axs,['Base','SARA']):panel(ax,data,name,i)
  if STYLE['show_title']:fig.suptitle(f'{data["display_id"]} | {data["title"]}',fontweight='bold')
  fig.subplots_adjust(left=.07,right=.97,bottom=.16,top=.83,wspace=.2)
  buf=io.BytesIO();fig.savefig(buf,format='png',dpi=100);plt.close(fig);buf.seek(0);frames.append(Image.open(buf).convert('RGB'))
 frames[0].save(out/f'case_{data["number"]:02d}.gif',save_all=True,append_images=frames[1:],duration=[500]*8+[1500],loop=0)

def diagnostics(data,out,formats):
 fig,axs=plt.subplots(1,3,figsize=(15,4.8));t=np.arange(9)*.5
 for name in ['Base','SARA']:
  m=data['dac_audit'][name];color=STYLE['colors'][name];axs[0].plot(t,m['speed_mps'],'-o',c=color,label=name,ms=3);axs[1].plot(t,m['outside_area_m2'],'-o',c=color,label=name,ms=3)
 for ax in axs[:2]:ax.set_xlabel('Time (s)');ax.set_xlim(0,4);ax.grid(alpha=.2);ax.legend()
 axs[0].set_ylabel('Speed magnitude (m/s)');axs[1].set_ylabel('Auxiliary body area outside road union (m²)')
 j=int(np.argmax(data['dac_audit']['SARA']['outside_area_m2']));panel(axs[2],data,'SARA',j);center=footprint(data['models']['SARA']['local_xy'][j],data['models']['SARA']['yaw_local'][j]).mean(0);axs[2].set_xlim(center[0]-4,center[0]+4);axs[2].set_ylim(center[1]-4,center[1]+4);axs[2].set_title(f'SARA detail at t={t[j]:g}s (same geometry)');fig.suptitle(data['display_id']+' | '+data['title']);fig.tight_layout()
 for fmt in formats:
  path=out/f'case_{data["number"]:02d}_diagnostics.{fmt}';fig.savefig(path,dpi=STYLE['dpi'])
  if fmt=='svg':path.write_text('\n'.join(l.rstrip() for l in path.read_text().splitlines())+'\n')
 plt.close(fig)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cases',type=int,nargs='+');p.add_argument('--gif',action='store_true');p.add_argument('--formats',nargs='+',default=['png','pdf','svg']);p.add_argument('--output',type=Path,default=HERE/'figures');args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
 plt.rcParams.update({'font.family':STYLE['font_family'],'font.size':STYLE['font_size'],'pdf.fonttype':42,'svg.fonttype':'none'})
 paths=[HERE/'data'/f'case_{n:02d}.json' for n in args.cases] if args.cases else sorted((HERE/'data').glob('case_*.json'))
 for path in paths:
  d=json.loads(path.read_text());snapshots(d,args.output,args.formats)
  diagnostics(d,args.output,args.formats)
  if args.gif:animate(d,args.output)
  print('Rendered',path.stem,flush=True)
if __name__=='__main__':main()
