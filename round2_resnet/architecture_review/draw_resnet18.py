"""Exact vector schematic; layout inspired by the user-supplied reference.
No images, training data, weights or model inference are used.
Run: python3 draw_resnet18.py
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, PathPatch, FancyArrowPatch
from matplotlib.path import Path as MPath
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':7,'pdf.fonttype':42,'svg.fonttype':'none'})
out=Path(__file__).resolve().parent
fig,ax=plt.subplots(figsize=(18/2.54,5.4/2.54))
fig.subplots_adjust(left=.008,right=.992,bottom=.015,top=.985)
ink='#28333b'; muted='#56636c'
colors=['#f0dfe0','#e4e1ee','#ece9cf','#f1e1ce']
y=1.85; bw=.45; bh=1.9

def text(x,y,s,**kw): ax.text(x,y,s,ha='center',va='center',color=ink,**kw)
def arrow(a,b,**kw):
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=5.4,lw=.65,color=ink,shrinkA=0,shrinkB=0,**kw))
def bar(x,label,c='#e8edf0'):
 ax.add_patch(Rectangle((x,y-bh/2),bw,bh,fc=c,ec=muted,lw=.65))
 text(x+bw/2,y,label,rotation=90,fontsize=6.2)
def arc(start,end,projection):
 top=y+2.0
 path=MPath([(start,y),(start+.18,top),(end-.22,top),(end,y+.1)], [MPath.MOVETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4])
 ax.add_patch(PathPatch(path,fill=False,color=ink,lw=.7,ls=(0,(3,2)) if projection else '-'))
 arrow((end,y+.19),(end,y+.085))
def addnode(x):
 ax.add_patch(Ellipse((x,y),.22,.14,fc='white',ec=ink,lw=.65,zorder=6))
 ax.plot([x-.052,x+.052],[y,y],color=ink,lw=.6,zorder=7)
 ax.plot([x,x],[y-.035,y+.035],color=ink,lw=.6,zorder=7)

bar(.1,'RGB input')
bar(1.05,'7×7, 64; s=2','#dce8dd')
bar(1.9,'3×3 pool; s=2','#e8edf0')
arrow((.55,y),(1.05,y));arrow((1.5,y),(1.9,y))
text(.2,.27,'3×128×128',fontsize=5.3)
text(1.85,.41,'Stem\n64×32×32',fontsize=5.3)
prev=2.35
stage_ranges=[]
x=3.05
for stage,(ch,side) in enumerate([(64,32),(128,16),(256,8),(512,4)]):
 left=x-.23
 for block in range(2):
  start=x-.23
  arrow((prev,y),(start,y))
  arrow((start,y),(x,y))
  stride=2 if stage>0 and block==0 else 1
  bar(x,f'3×3, {ch}' + ('; s=2' if stride==2 else ''),colors[stage])
  bar(x+.77,f'3×3, {ch}',colors[stage])
  arrow((x+bw,y),(x+.77,y))
  node=x+1.52
  arrow((x+.77+bw,y),(node-.1,y))
  arc(start,node,stage>0 and block==0)
  addnode(node)
  if stage>0 and block==0:
   text((start+node)/2,3.48,'1×1, s=2',fontsize=6)
  prev=node+.1
  x+=2.15
 right=prev+.1
 stage_ranges.append((left,right))
 ax.plot([left,left,right,right],[.64,.55,.55,.64],color=muted,lw=.55)
 text((left+right)/2,.27,f'Layer {stage+1} · {ch} channels\n{ch}×{side}×{side}',fontsize=6.8)
 x+=.25

x=prev+.7
bar(x,'Avg pool')
arrow((prev,y),(x,y))
text(x+bw/2,.27,'512×1×1',fontsize=5.3)
x2=x+1.5
bar(x2,'FC 512 → 3','#ead4b9')
arrow((x+bw,y),(x2,y))
text(x2+bw/2,.27,'3 logits',fontsize=5.3)
text((x+bw+x2)/2,2.94,'flatten\n512',fontsize=6)
freeze_end=(stage_ranges[2][1]+stage_ranges[3][0])/2
ax.plot([1.05,freeze_end],[3.95,3.95],color=muted,lw=.65)
text((1.05+freeze_end)/2,4.13,'Frozen: stem and layers 1–3',fontsize=7)
ax.plot([freeze_end+.2,x2+bw],[3.95,3.95],color='#9b6937',lw=.65)
text((freeze_end+.2+x2+bw)/2,4.13,'Trainable: layer 4 and classifier',fontsize=7)
ax.set_xlim(-.65,x2+bw+.45);ax.set_ylim(-.06,4.37);ax.axis('off')
for ext in ['svg','pdf','png']:
 fig.savefig(out/f'resnet18_final.{ext}',dpi=300,facecolor='white')
plt.close(fig)
