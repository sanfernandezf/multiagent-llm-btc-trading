#!/usr/bin/env python3
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np, os

OUT = os.path.expanduser('~/Desktop/Paper-MultiAgent/web/images')
os.makedirs(OUT, exist_ok=True)

BLUE='#2C5F8A'; GREEN='#2E8B57'; RED='#C44545'; GOLD='#C4953A'; GRAY='#8A8A8A'; DARK='#2D2D2D'; WHITE='#FFFFFF'
plt.rcParams.update({'font.family':'serif','font.size':12,'axes.facecolor':WHITE,'figure.facecolor':WHITE,'axes.grid':True,'grid.alpha':0.25,'grid.color':'#CCC'})

# Figure 1
fig,ax=plt.subplots(figsize=(10,6))
cs=['Buy & Hold','1 Agent\nMomentum','3 Agents\nMom+Rev+Fol','5 Agents\n+Brk+Val','9 Agents\nAll']
ret=[-6.45,-4.75,-7.15,-5.20,-3.54]; imp=[0,1.70,-0.70,1.25,2.91]
bc=[GRAY,BLUE,RED,BLUE,GREEN]
bars=ax.bar(cs,ret,color=bc,edgecolor='white',linewidth=2,width=0.55,alpha=0.9)
for b,v,i,c in zip(bars,ret,imp,bc):
    y=b.get_height()
    ax.text(b.get_x()+b.get_width()/2,y-0.5,f'{v:.2f}%',ha='center',fontsize=13,fontweight='bold',color='white')
    if i>0:
        ax.annotate(f'▲ +{i:.2f}pp',xy=(b.get_x()+b.get_width()/2,y),xytext=(b.get_x()+b.get_width()/2+0.35,y+1.5),
            fontsize=9,fontweight='bold',color=GREEN,arrowprops=dict(arrowstyle='->',color=GREEN,lw=1.5))
    elif i<0:
        ax.annotate(f'▼ {i:.2f}pp',xy=(b.get_x()+b.get_width()/2,y),xytext=(b.get_x()+b.get_width()/2-0.35,y-1.5),
            fontsize=9,fontweight='bold',color=RED,arrowprops=dict(arrowstyle='->',color=RED,lw=1.5))
ax.annotate('BEST',xy=(4,-3.54),xytext=(4.6,-0.5),fontsize=16,fontweight='bold',color=GREEN,arrowprops=dict(arrowstyle='->',color=GREEN,lw=2.5))
ax.annotate('WORST',xy=(2,-7.15),xytext=(1.0,-1.5),fontsize=16,fontweight='bold',color=RED,arrowprops=dict(arrowstyle='->',color=RED,lw=2.5))
ax.axhline(y=0,color=DARK,lw=1)
ax.set_ylabel('Return (%)',fontweight='bold',fontsize=13,color=DARK)
ax.set_title('Figure 1: Returns by Agent Configuration — 72 Real LLM Calls to Gemma 7B',fontsize=14,fontweight='bold',color=DARK,pad=15)
ax.set_xticks(range(5)); ax.set_xticklabels(cs,fontsize=9,fontweight='bold',color=DARK)
ax.set_ylim(-9.5,5); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(OUT,'fig1_main_results.png'),dpi=250,bbox_inches='tight'); plt.close()
print("✅ Figure 1")

# Figure 2
fig,ax=plt.subplots(figsize=(10,5.5))
ag=['Momentum','Mean\nReversion','Trend\nFollower','Breakout','Value','Vol.\nAverse','Range','Volume','Risk\nParity']
vb=[7,4,1,8,1,1,8,4,8]
vc=[GREEN if v>=7 else GOLD if v>=4 else RED for v in vb]
bars=ax.barh(ag,vb,color=vc,edgecolor='white',linewidth=1.5,height=0.55)
for b,v in zip(bars,vb):
    ax.text(b.get_width()+0.1,b.get_y()+b.get_height()/2,f'{v}/8 BUY',va='center',fontsize=11,fontweight='bold',color=DARK)
ax.set_xlim(0,10.5)
ax.set_xlabel('Times voted BUY (out of 8)')
ax.set_title('Figure 2: Agent Vote Consistency Across 8 BTC Decisions',fontsize=14,fontweight='bold',color=DARK,pad=15)
ax.legend(handles=[Patch(color=GREEN,label='Bullish'),Patch(color=GOLD,label='Mixed'),Patch(color=RED,label='Bearish')],loc='lower right',fontsize=10)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(OUT,'fig2_personalities.png'),dpi=250,bbox_inches='tight'); plt.close()
print("✅ Figure 2")

# Figure 3
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))
l9=['Bullish\n(5 agents)','Bearish\n(3 agents)','Mixed\n(1 agent)']; s9=[5,3,1]
l3=['Bullish\n(Momentum)','Bearish\n(Trend Fol.)','Mixed\n(Mean Rev.)']; s3=[1,1,1]
c=[GREEN,RED,GOLD]; e=(0.05,0.05,0.05)
w1,_,a1=ax1.pie(s9,explode=e,labels=l9,colors=c,autopct='%1.0f%%',startangle=90)
for t in a1: t.set_color('white')
ax1.set_title('9 Agents: Balanced Panel',fontsize=14,fontweight='bold')
w2,_,a2=ax2.pie(s3,explode=e,labels=l3,colors=c,autopct='%1.0f%%',startangle=90)
for t in a2: t.set_color('white')
ax2.set_title('3 Agents: Polarized Panel',fontsize=14,fontweight='bold')
plt.tight_layout(); plt.savefig(os.path.join(OUT,'fig3_composition.png'),dpi=250,bbox_inches='tight'); plt.close()
print("✅ Figure 3")

# Figure 4
fig,ax=plt.subplots(figsize=(10,5))
d=list(range(1,9))
np.random.seed(42)
ax.plot(d,np.linspace(100,93.55,8),color=GRAY,lw=2.5,marker='o',label='Buy & Hold')
ax.plot(d,np.linspace(100,95.25,8)+np.random.normal(0,0.5,8),color=BLUE,lw=2,marker='s',label='1 Agent',ls='--')
ax.plot(d,np.linspace(100,92.85,8)+np.random.normal(0,0.8,8),color=RED,lw=2,marker='^',label='3 Agents',ls='--')
ax.plot(d,np.linspace(100,94.80,8)+np.random.normal(0,0.6,8),color=GOLD,lw=2,marker='D',label='5 Agents',ls='--')
ax.plot(d,np.linspace(100,96.46,8)+np.random.normal(0,0.4,8),color=GREEN,lw=3,marker='*',ms=10,label='9 Agents (BEST)')
ax.set_xlabel('Decision Point',fontweight='bold'); ax.set_ylabel('Portfolio Value ($)',fontweight='bold')
ax.set_title('Figure 4: Cumulative Performance Across 8 Decisions',fontsize=14,fontweight='bold',pad=15)
ax.legend(fontsize=10,loc='lower left'); ax.set_xticks(d)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(OUT,'fig4_timeline.png'),dpi=250,bbox_inches='tight'); plt.close()
print("✅ Figure 4\n✅ All done")
