"""独立绘图入口：python plot_cases.py --cases M01 M03 D01 D02 D03 D06 --gif"""
import argparse,importlib.util,json,copy
from pathlib import Path
HERE=Path(__file__).resolve().parent
# 可在这里统一改图；原始模型轨迹保存在M/data与D/data。
STYLE={'font_size':11,'font_family':'DejaVu Sans','dpi':200,'figsize':[12,8],
       'colors':{'Base':'#cc4c3b','SARA':'#197bbd'},'trajectory_linewidth':2,
       'marker_size':3,'show_start_body':True,'show_title':True}
TIMES=[2.,4.]
CASE_OVERRIDES={}  # 例：{'D02': {'xlim': [-8,32], 'ylim': [-8,20]}}
M_STYLE={'highlight_actor':True,'highlight_color':'#e6aa3c'}
D_STYLE={}
SELECTED=['M01','M03','D01','D02','D03','D06']

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cases',nargs='+',default=SELECTED);p.add_argument('--gif',action='store_true');p.add_argument('--formats',nargs='+',choices=['png','pdf','svg'],default=['png','pdf','svg']);p.add_argument('--output',type=Path,default=HERE/'edited_figures');args=p.parse_args();cases=[]
 for raw in args.cases:
  raw=raw.upper();group=raw[:1]
  if group not in ['M','D'] or not raw[1:].isdigit():p.error('编号格式为 M01、M03、D01、D02、D03、D06')
  fid=group+str(int(raw[1:])).zfill(2)
  if fid not in SELECTED:p.error('本包没有 '+fid)
  cases.append(fid)
 modules={}
 for fid in cases:
  group=fid[0];n=int(fid[1:])
  if group not in modules:
   spec=importlib.util.spec_from_file_location('_sara_selected_'+group,HERE/group/'plot_cases.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);modules[group]=mod
  mod=modules[group];mod.STYLE.update(copy.deepcopy(STYLE));mod.STYLE.update(copy.deepcopy(M_STYLE if group=='M' else D_STYLE));mod.STYLE['case_overrides']={str(int(k[1:])):v for k,v in CASE_OVERRIDES.items() if k.startswith(group)};mod.TIMES[:]=TIMES;mod.plt.rcParams.update({'font.family':mod.STYLE['font_family'],'font.size':mod.STYLE['font_size'],'pdf.fonttype':42,'svg.fonttype':'none'});out=args.output/group;out.mkdir(parents=True,exist_ok=True);d=json.loads((HERE/group/'data'/f'case_{n:02d}.json').read_text());mod.snapshots(d,out,args.formats);mod.diagnostics(d,out,args.formats)
  if group=='M' and d.get('critical_times'):
   oldtimes=mod.TIMES[:];oldsize=mod.STYLE['figsize'];oldstem=d['file_stem'];mod.TIMES[:]=d['critical_times'];mod.STYLE['figsize']=[22,8];d['file_stem']=oldstem+'_critical';mod.snapshots(d,out,args.formats);mod.TIMES[:]=oldtimes;mod.STYLE['figsize']=oldsize;d['file_stem']=oldstem
  if args.gif:mod.animate(d,out)
  print('Rendered',fid,'->',out,flush=True)
if __name__=='__main__':main()
