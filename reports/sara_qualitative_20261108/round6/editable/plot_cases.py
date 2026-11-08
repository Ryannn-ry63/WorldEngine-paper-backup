"""Portable paper plots: Python + NumPy + Matplotlib; no simulator or map SDK."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

HERE = Path(__file__).resolve().parent


def footprint(xy, theta):
    # Pacifica body: length 5.176 m, width 2.297 m; center 1.461 m ahead of rear axle.
    corners = np.array([[2.588,1.1485],[2.588,-1.1485],[-2.588,-1.1485],[-2.588,1.1485]])
    corners[:,0] += 1.461
    c,s = np.cos(theta),np.sin(theta)
    return corners @ np.array([[c,s],[-s,c]]) + xy


def draw_map(ax, data, style):
    for shape in data['map']:
        layer = shape['layer']
        if layer in style['hidden_map_layers']:
            continue
        xy = np.array(shape['xy'])
        if shape['type'] == 'Polygon':
            color = style['road_fill'] if layer != 'crosswalks' else style['crosswalk_fill']
            ax.add_patch(Polygon(xy,facecolor=color,edgecolor=style['map_edge'],lw=style['map_linewidth'],zorder=0))
        else:
            ax.plot(xy[:,0],xy[:,1],color=style['map_edge'],lw=style['map_linewidth'],ls='--' if layer == 'baseline_paths' else '-',zorder=1)


def draw_case(data, style, out, formats):
    number = str(data['number'])
    custom = style.get('case_overrides',{}).get(number,{})
    models = data['models']
    allxy = np.concatenate([m['local_xy'] for m in models.values()])
    lo,hi = allxy.min(0)-7,allxy.max(0)+8
    lo[1],hi[1] = min(lo[1],-9),max(hi[1],9)
    xlim = custom.get('xlim',[float(lo[0]),float(hi[0])])
    ylim = custom.get('ylim',[float(lo[1]),float(hi[1])])
    speed_panel = style['show_speed']
    fig,axs = plt.subplots(1,3 if speed_panel else 2,
        figsize=style['figsize_with_speed'] if speed_panel else style['figsize_bev'],squeeze=False)
    axs = axs[0]
    for ax,name in zip(axs,('Base','SARA')):
        draw_map(ax,data,style)
        m = models[name]; xy = np.array(m['local_xy']); color = style['colors'][name]
        for i in style['actor_frames']:
            for polygon in m['actors'][i]:
                p = np.array(polygon); center=p.mean(0)
                if not (xlim[0]-5 < center[0] < xlim[1]+5 and ylim[0]-5 < center[1] < ylim[1]+5):
                    continue
                ax.add_patch(Polygon(p,fc=style['actor_fill'],ec=style['actor_edge'],alpha=.16 if i == 0 else .5,lw=.5,zorder=2))
        ax.plot(xy[:,0],xy[:,1],'-o',c=color,lw=style['trajectory_linewidth'],ms=style['marker_size'],zorder=4)
        if style['show_footprints']:
            for i in range(0,len(xy),style['footprint_stride']):
                ax.add_patch(Polygon(footprint(xy[i],m['yaw_local'][i]),fc=color,ec=color,alpha=.12+.035*i,lw=.7,zorder=3))
        ax.scatter(*xy[0],marker='s',s=28,c='black',zorder=6)
        score = data['metrics'][name][data['mode']]['score']*100
        title = style['model_labels'][name]
        if style['show_scores']: title += f'   PDM {score:.1f}'
        ax.set_title(title,fontweight='bold')
        ax.set(xlim=xlim,ylim=ylim,xlabel=style['x_label'],ylabel=style['y_label'])
        ax.set_aspect('equal'); ax.spines[['top','right']].set_visible(False)
    if speed_panel:
        ax=axs[2]
        for name,m in models.items():
            ax.plot(m['time_s'],m['speed_mps'],'-o',c=style['colors'][name],label=style['model_labels'][name],ms=style['marker_size'],lw=style['trajectory_linewidth'])
        ax.set(xlabel='Closed-loop time (s)',ylabel='Speed (m/s)',xlim=(0,4))
        ax.grid(alpha=.2); ax.legend(loc='upper right'); ax.spines[['top','right']].set_visible(False)
        if style['show_metrics']:
            lines=[f"{data['mode']} metrics (0–100)",'          Base     SARA']
            for label,key in [('NC','no_at_fault_collisions'),('DAC','drivable_area_compliance'),('TTC','time_to_collision_within_bound'),('EP','ego_progress')]:
                lines.append(f'{label:<5} {data["metrics"]["Base"][data["mode"]][key]*100:8.1f} {data["metrics"]["SARA"][data["mode"]][key]*100:8.1f}')
            ax.text(.03,.03,'\n'.join(lines),transform=ax.transAxes,fontfamily='monospace',fontsize=style['font_size']*.8,bbox={'fc':'white','alpha':.85,'ec':'#dddddd'})
    if style['show_title']:
        fig.suptitle(custom.get('title',f"{data['number']:02d} | {data['title']}"),fontweight='bold')
    if style['show_footer']:
        times=', '.join(f"{models['Base']['time_s'][i]:g}" for i in style['actor_frames'])
        footstep=.5*style['footprint_stride']
        footer=f'Executed 4 s closed-loop path; samples every 0.5 s. Gray actors: t={times} s.'
        if style['show_footprints']: footer+=f' Ego footprints every {footstep:g} s.'
        fig.text(.5,.015,footer,ha='center',fontsize=style['font_size']*.75)
    fig.tight_layout(rect=(0,.055 if style['show_footer'] else 0,1,.93 if style['show_title'] else 1))
    for fmt in formats:
        target = out/f"case_{data['number']:02d}.{fmt}"
        fig.savefig(target,dpi=style['dpi'])
        if fmt == "svg":
            target.write_text("\n".join(line.rstrip() for line in target.read_text().splitlines())+"\n")
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',type=int,nargs='+',help='Default: all cases in data/')
    parser.add_argument('--style',type=Path,default=HERE/'style.json')
    parser.add_argument('--output',type=Path,default=HERE/'figures')
    parser.add_argument('--formats',nargs='+',choices=['png','pdf','svg'],default=['png','pdf','svg'])
    args=parser.parse_args()
    style=json.loads(args.style.read_text())
    if style['footprint_stride'] < 1: parser.error('footprint_stride must be positive')
    if any(i not in range(9) for i in style['actor_frames']): parser.error('actor_frames must be indices 0 through 8')
    plt.rcParams.update({'font.family':style['font_family'],'font.size':style['font_size'],
        'axes.titlesize':style['font_size']+1,'figure.titlesize':style['font_size']+2,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    args.output.mkdir(parents=True,exist_ok=True)
    paths=[HERE/'data'/f'case_{n:02d}.json' for n in args.cases] if args.cases else sorted((HERE/'data').glob('case_*.json'))
    for path in paths:
        draw_case(json.loads(path.read_text()),style,args.output,args.formats)
        print('Rendered',path.stem,flush=True)


if __name__ == '__main__':
    main()
