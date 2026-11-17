"""Optional shareable evaluation figure, independent of the web interface."""
from pathlib import Path
import os,json
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
def main():
    r=json.loads((ROOT/'reports/evaluation.json').read_text(encoding='utf-8'));samples=json.loads((ROOT/'dist/data/samples.json').read_text(encoding='utf-8'));k=r['selected'];m=r['models'][k]['test'];c=np.array(m['confusion'])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(12,10),facecolor='#f8f7fc');gs=fig.add_gridspec(3,6,height_ratios=[1,1,.75],hspace=.55,wspace=.8)
    ax=fig.add_subplot(gs[:2,:3]);ax.imshow(c,cmap='Purples',vmin=0,vmax=c.max());ax.set_xticks(range(10));ax.set_yticks(range(10));ax.set(xlabel='Predicted digit',ylabel='Reference digit',title='Official different-writer test / confusion counts')
    for i in range(10):
        for j in range(10):
            if c[i,j]:ax.text(j,i,str(c[i,j]),ha='center',va='center',fontsize=8,color='white' if c[i,j]>c.max()/2 else '#694181')
    ax=fig.add_subplot(gs[0,3:]);keys=list(r['models']);x=np.arange(3);ax.bar(x-.17,[r['models'][n]['validation']['accuracy']*100 for n in keys],width=.34,color='#c5b5dc',label='Validation');ax.bar(x+.17,[r['models'][n]['test']['accuracy']*100 for n in keys],width=.34,color='#8062a7',label='Official test');ax.set_xticks(x,['Linear','3-NN','Neural']);ax.set(ylabel='Accuracy (%)',ylim=(0,105),title='Fixed candidates / no test-based selection');ax.legend(frameon=False,fontsize=8,ncol=2,loc='lower left');ax.grid(axis='y',alpha=.15)
    ax=fig.add_subplot(gs[1,3:]);stress=r['robustness'];ax.barh([s['label'] for s in stress[::-1]],[s['accuracy']*100 for s in stress[::-1]],color=['#c6b1d7']*5+['#9bac7d']);ax.set(xlabel='Official-test accuracy (%)',xlim=(0,105),title='The same model under small input changes');ax.tick_params(axis='y',labelsize=8);ax.grid(axis='x',alpha=.15)
    mistakes=[s for s in samples if s['label']!=s['predictions'][k]][:6]
    for i,s in enumerate(mistakes):
        ax=fig.add_subplot(gs[2,i]);ax.imshow(np.array(s['pixels']).reshape(8,8),cmap='Purples',vmin=0,vmax=16,interpolation='nearest');ax.set_xticks([]);ax.set_yticks([]);ax.set_title(f"#{s['id']+1}: {s['label']} → {s['predictions'][k]}",fontsize=9);ax.set_xlabel(f"Score {s['scores'][k]*100:.0f}%",fontsize=8)
    fig.suptitle('InkLab / Small images, visible failure modes',x=.07,y=.98,ha='left',fontsize=20,fontweight='bold',color='#44324f');fig.text(.07,.94,'3-nearest neighbors: 1,758 / 1,797 correct (97.83%). The first six errors in test-file order are shown below.',fontsize=10,color='#816b92');fig.subplots_adjust(left=.07,right=.97,top=.88,bottom=.13)
    fig.text(.07,.045,'Source: Alpaydin & Kaynak (1998), UCI Optdigits, DOI 10.24432/C50P49 / CC BY 4.0.\nModel scores are uncalibrated; doodles bypass original preprocessing. Neural training reached its fixed iteration cap.',fontsize=9,color='#816b92')
    fig.axes[0].set_position([.07,.37,.40,.49])
    fig.axes[1].set_position([.63,.69,.34,.19])
    fig.axes[2].set_position([.63,.38,.34,.19])
    fig.savefig(ROOT/'reports/evaluation.png',dpi=150,facecolor=fig.get_facecolor());plt.close(fig)
if __name__=='__main__':main()
