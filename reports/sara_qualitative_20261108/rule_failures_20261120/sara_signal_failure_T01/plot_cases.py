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
STYLE={'colors':{'Base':'#cc4c3b','SARA':'#197bbd'},'font_size':11,'font_family':'DejaVu Sans','dpi':200,'figsize':[12,8], 'road_fill':'#eef0f2','map_edge':'#c2c7cc','actor_fill':'#89939c','actor_edge':'#59636e','highlight_actor':True,'highlight_color':'#e6aa3c','trajectory_linewidth':2,'marker_size':3,'show_start_body':True,'show_title':True,'case_overrides':{}}
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
 area=np.array(data['stop_area_xy']);ax.add_patch(Polygon(area,fc='#f4bbbb',ec='#bf2222',lw=1.1,alpha=.4,hatch='//',zorder=2));edge=np.array(data['stop_exit_edge_xy']);ax.plot(edge[:,0],edge[:,1],c='#a91414',lw=2,zorder=4)
 m=data['models'][name];xy=np.array(m['local_xy']);color=STYLE['colors'][name]
 for j,p in enumerate(m['actors'][i]):
  p=np.array(p);c=p.mean(0)
  if xlim[0]-6<c[0]<xlim[1]+6 and ylim[0]-6<c[1]<ylim[1]+6:
   focus=STYLE['highlight_actor'] and j==m.get('highlight_actor_index',[None]*9)[i]
   ax.add_patch(Polygon(p,fc=STYLE['highlight_color'] if focus else STYLE['actor_fill'],ec='#8b5c12' if focus else STYLE['actor_edge'],alpha=.8,lw=1.1 if focus else .6,zorder=3))
   if focus:ax.add_patch(Polygon(p,fc='none',ec='#8b5c12',lw=1.1,zorder=8))
 if STYLE['show_start_body']:ax.add_patch(Polygon(footprint(xy[0],m['yaw_local'][0]),fc='none',ec=color,ls=':',lw=1,zorder=4))
 ax.plot(xy[:i+1,0],xy[:i+1,1],'-o',c=color,lw=STYLE['trajectory_linewidth'],ms=STYLE['marker_size'],zorder=5)
 ax.add_patch(Polygon(footprint(xy[i],m['yaw_local'][i]),fc=color,ec='#222222',alpha=.82,lw=.8,zorder=6))
 ax.scatter(*xy[0],s=16,c='black',marker='s',zorder=7)
 ax.set(xlim=xlim,ylim=ylim,xlabel='Initial-heading longitudinal axis (m)',ylabel='Lateral axis (m)',title=f'{name} | t = {m["time_s"][i]:g} s | v = {m["speed_mps"][i]:.1f} m/s | '+('RED' if data['signal_audit']['events'][name][i]['all_known_controls_red'] else 'signal unobserved'));ax.set_aspect('equal');ax.spines[['top','right']].set_visible(False)

def snapshots(data,out,formats):
 fig,axs=plt.subplots(2,len(TIMES),figsize=STYLE['figsize'],squeeze=False)
 for row,name in enumerate(['Base','SARA']):
  for col,t in enumerate(TIMES):
   times=np.array(data['models'][name]['time_s']);assert np.any(np.isclose(times,t));i=int(np.argmin(abs(times-t)));panel(axs[row,col],data,name,i)
 if STYLE['show_title']:fig.suptitle(f'{data.get("display_id","")} | {data.get("title",data["scene"].split("-")[-1])}',fontweight='bold')
 footer='Fixed initial coordinates. Red hatch: mapped signal stop area. Thick red line: its downstream boundary.'
 if any('highlight_actor_index' in m for m in data['models'].values()):footer+=' Orange: same tracked vehicle.'
 fig.text(.5,.025,footer,ha='center',fontsize=9)
 fig.subplots_adjust(left=.07,right=.97,bottom=.1,top=.91,hspace=.32,wspace=.2)
 stem=data.get('file_stem',f'case_{data["number"]:02d}')
 for fmt in formats:
  path=out/f'{stem}_2s_4s.{fmt}';fig.savefig(path,dpi=STYLE['dpi'])
  if fmt=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
 plt.close(fig)

def animate(data,out):
 frames=[]
 for i in range(9):
  fig,axs=plt.subplots(1,2,figsize=(12,4.6))
  for ax,name in zip(axs,['Base','SARA']):panel(ax,data,name,i)
  if STYLE['show_title']:fig.suptitle(f'{data["display_id"]} | {data["title"]}',fontweight='bold')
  fig.subplots_adjust(left=.07,right=.97,bottom=.16,top=.83,wspace=.2)
  buf=io.BytesIO();fig.savefig(buf,format='png',dpi=100);plt.close(fig);buf.seek(0);frames.append(Image.open(buf).convert('RGB'))
 stem=data.get('file_stem',f'case_{data["number"]:02d}')
 frames[0].save(out/f'{stem}.gif',save_all=True,append_images=frames[1:],duration=[500]*8+[1500],loop=0)

def diagnostics(data,out,formats):
 fig,axs=plt.subplots(1,3,figsize=(14,4));t=np.arange(9)*.5
 for name,m in data['models'].items():
  c=STYLE['colors'][name];axs[0].plot(t,m['speed_mps'],'-o',c=c,label=name,ms=3);ev=data['signal_audit']['events'][name];axs[1].plot(t,[x['front_after_exit_m'] for x in ev],'-o',c=c,label=name,ms=3)
 axs[0].set_ylabel('Recorded speed magnitude (m/s)');axs[1].set_ylabel('Front past downstream stop-area edge (m)');axs[1].axhline(0,c='#b22',ls='--',lw=1)
 keys=data['signal_audit']['control_connectors'];y=[1 if all(r['controlled_status'][k]=='red' for k in keys) else -1 if any(r['controlled_status'][k]=='unobserved' for k in keys) else 0 for r in data['database_signal_evidence']['samples']]
 axs[2].step(t,y,where='post',label='All 3 controlled signals',c='#b22222',lw=2)
 axs[2].set_yticks([-1,0,1],['Not recorded','Green','Red']);axs[2].set_ylim(-1.2,1.2);axs[2].set_ylabel('Original DB: controlled signal status')
 for ax in axs:ax.set_xlabel('Time (s)');ax.set_xlim(0,4);ax.grid(alpha=.2);ax.legend(fontsize=8)
 fig.suptitle(data['display_id']+' | Signal and motion verification');fig.tight_layout()
 for fmt in formats:
  path=out/(data['file_stem']+'_diagnostics.'+fmt);fig.savefig(path,dpi=STYLE['dpi'])
  if fmt=='svg':path.write_text('\n'.join(l.rstrip() for l in path.read_text().splitlines())+'\n')
 plt.close(fig)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cases',type=int,nargs='+');p.add_argument('--gif',action='store_true');p.add_argument('--formats',nargs='+',default=['png','pdf','svg']);p.add_argument('--output',type=Path,default=HERE/'figures');args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
 plt.rcParams.update({'font.family':STYLE['font_family'],'font.size':STYLE['font_size'],'pdf.fonttype':42,'svg.fonttype':'none'})
 paths=[HERE/'data'/f'case_{n:02d}.json' for n in args.cases] if args.cases else sorted((HERE/'data').glob('case_*.json'))
 for path in paths:
  d=json.loads(path.read_text());snapshots(d,args.output,args.formats)
  diagnostics(d,args.output,args.formats)
  if d.get('critical_times'):
   ts=TIMES[:];sz=STYLE['figsize'];stem=d['file_stem'];TIMES[:]=d['critical_times'];STYLE['figsize']=[22,8];d['file_stem']=stem+'_critical';snapshots(d,args.output,args.formats);TIMES[:]=ts;STYLE['figsize']=sz;d['file_stem']=stem
  if args.gif:animate(d,args.output)
  print('Rendered',path.stem,flush=True)
if __name__=='__main__':main()
