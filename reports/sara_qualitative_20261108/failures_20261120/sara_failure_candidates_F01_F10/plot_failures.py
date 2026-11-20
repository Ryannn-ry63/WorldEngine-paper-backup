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
STYLE={'colors':{'Base':'#cc4c3b','SARA':'#197bbd'},'font_size':11,'font_family':'DejaVu Sans','dpi':200,'figsize':[12,8], 'road_fill':'#eef0f2','map_edge':'#c2c7cc','actor_fill':'#89939c','actor_edge':'#59636e','highlight_actor':True,'show_reference':True,'show_failure_regions':True,'highlight_color':'#e6aa3c','trajectory_linewidth':2,'marker_size':3,'show_start_body':True,'show_title':True,'case_overrides':{'8':{'xlim':[-6,28],'ylim':[-15,10]}}}
TIMES=[2.,4.]

def footprint(xy,theta):
 p=np.array([[2.588,1.1485],[2.588,-1.1485],[-2.588,-1.1485],[-2.588,1.1485]]);p[:,0]+=1.461;c,s=np.cos(theta),np.sin(theta)
 return p@np.array([[c,s],[-s,c]])+xy

def limits(data):
 xy=np.concatenate([m['local_xy'] for m in data['models'].values()])
 if data.get('show_reference') and STYLE['show_reference']:
  r=data['logged_reference'];mask=np.array(r['time_s'])<=4.;xy=np.concatenate([xy,np.array(r['local_xy'])[mask]])
 lo=xy.min(0)-8;hi=xy.max(0)+10;lo[1]=min(lo[1],-9);hi[1]=max(hi[1],9)
 custom=STYLE['case_overrides'].get(str(data.get('number','')),{})
 return custom.get('xlim',[lo[0],hi[0]]),custom.get('ylim',[lo[1],hi[1]])

def panel(ax,data,name,i):
 xlim,ylim=limits(data)
 for shape in data['map']:
  p=np.array(shape['xy'])
  if shape['type']=='Polygon':ax.add_patch(Polygon(p,fc=STYLE['road_fill'] if shape['layer']!='crosswalks' else '#dce0e3',ec=STYLE['map_edge'],lw=.5,zorder=0))
  else:ax.plot(p[:,0],p[:,1],c=STYLE['map_edge'],lw=.6,ls='--' if shape['layer']=='baseline_paths' else '-',zorder=1)
 if STYLE['show_failure_regions']:
  for region in data.get('failure_regions',[]):ax.add_patch(Polygon(region['xy'],fc='#e5b9c3',ec='#ba7083',alpha=.35,hatch='///',lw=.5,zorder=2))
 if data.get('show_reference') and STYLE['show_reference']:
  r=data['logged_reference'];mask=np.array(r['time_s'])<=4.;p=np.array(r['local_xy'])[mask];ax.plot(p[:,0],p[:,1],ls='--',c='#45936a',lw=1.4,zorder=2)
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
 ax.set(xlim=xlim,ylim=ylim,xlabel='Initial-heading longitudinal axis (m)',ylabel='Lateral axis (m)',title=f'{name} | t = {m["time_s"][i]:g} s');ax.set_aspect('equal');ax.spines[['top','right']].set_visible(False)

def snapshots(data,out,formats):
 fig,axs=plt.subplots(2,len(TIMES),figsize=STYLE['figsize'],squeeze=False)
 for row,name in enumerate(['Base','SARA']):
  for col,t in enumerate(TIMES):
   times=np.array(data['models'][name]['time_s']);assert np.any(np.isclose(times,t));i=int(np.argmin(abs(times-t)));panel(axs[row,col],data,name,i)
 if STYLE['show_title']:fig.suptitle(f'{data.get("display_id","")} | {data.get("title",data["scene"].split("-")[-1])}',fontweight='bold')
 notes=['Fixed coordinates; dotted body: initial pose.']
 if 'interaction_audit' in data and STYLE['highlight_actor']:notes.append('Orange: same tracked vehicle.')
 if data.get('failure_regions') and STYLE['show_failure_regions']:notes.append('Hatched: opposing lane.')
 if data.get('show_reference') and STYLE['show_reference']:notes.append('Green dashed: logged human path (0–4 s, context).')
 fig.text(.5,.025,' '.join(notes),ha='center',fontsize=8)
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
 fig,axs=plt.subplots(1,2,figsize=(11,3.8));kind=data['diagnostic']
 for name,m in data['models'].items():
  t=np.array(m['time_s']);c=STYLE['colors'][name];axs[0].plot(t,m['speed_mps'],'-o',c=c,label=name,ms=3)
  if kind=='opposite':y=[s['opposite_lane_body_fraction']*100 for s in data['direction_audit'][name]['samples']];label='Body area in opposing lane (%)'
  elif kind=='coverage':y=[s['aligned_body_fraction']*100 for s in data['direction_audit'][name]['samples']];label='Aligned-lane body coverage (%)'
  elif kind=='heading':y=np.rad2deg(np.unwrap(m['yaw_local']));label='Heading relative to start (degrees)'
  elif kind=='interaction':y=data['interaction_audit']['models'][name]['target_overlap_m2'];label='Overlap with tracked vehicle (m²)'
  else:
   xy=np.array(m['local_xy']);th=np.unwrap(m['yaw_local']);mid=(th[1:]+th[:-1])/2;y=(np.diff(xy,axis=0)*np.c_[np.cos(mid),np.sin(mid)]).sum(1);t=t[1:];label='Signed forward step per 0.5 s (m)'
  axs[1].plot(t,y,'-o',c=c,label=name,ms=3)
 if kind=='heading':
  r=data['logged_reference'];t=np.array(r['time_s']);mask=t<=4;axs[1].plot(t[mask],np.rad2deg(np.array(r['yaw_local'])[mask]),'--',c='#45936a',label='Logged human reference')
 axs[1].axhline(0,color='#777',lw=.7);axs[0].set_ylabel('Recorded speed magnitude (m/s)');axs[1].set_ylabel(label)
 for ax in axs:ax.set_xlabel('Time (s)');ax.set_xlim(0,4);ax.grid(alpha=.2);ax.legend(fontsize=8);ax.spines[['top','right']].set_visible(False)
 fig.suptitle(data['display_id']+' | '+data['title'],fontsize=12);fig.tight_layout()
 for fmt in formats:
  path=out/(data['file_stem']+'_diagnostics.'+fmt);fig.savefig(path,dpi=STYLE['dpi'])
  if fmt=='svg':path.write_text('\n'.join(x.rstrip() for x in path.read_text().splitlines())+'\n')
 plt.close(fig)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cases',nargs='+');p.add_argument('--gif',action='store_true');p.add_argument('--formats',nargs='+',default=['png','pdf','svg']);p.add_argument('--output',type=Path,default=HERE/'figures');args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
 plt.rcParams.update({'font.family':STYLE['font_family'],'font.size':STYLE['font_size'],'pdf.fonttype':42,'svg.fonttype':'none'})
 paths=[HERE/'data'/('failure_F'+str(int(n.upper().lstrip('F'))).zfill(2)+'.json') for n in args.cases] if args.cases else sorted((HERE/'data').glob('failure_F*.json'))
 for path in paths:
  d=json.loads(path.read_text());snapshots(d,args.output,args.formats)
  diagnostics(d,args.output,args.formats)
  if d.get('critical_times'):
   old=TIMES[:];oldsize=STYLE['figsize'];oldstem=d['file_stem'];TIMES[:]=d['critical_times'];STYLE['figsize']=[18,8];d['file_stem']=oldstem+'_critical';snapshots(d,args.output,args.formats);TIMES[:]=old;STYLE['figsize']=oldsize;d['file_stem']=oldstem
  if args.gif:animate(d,args.output)
  print('Rendered',path.stem,flush=True)
if __name__=='__main__':main()
