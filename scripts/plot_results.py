"""Regenerate pilot figures from measured CSV only."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot(run, output):
    frame=pd.read_csv(run/'metrics.csv')
    output.mkdir(parents=True,exist_ok=True)
    for fraction,part in frame.groupby('fraction'):
        fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
        for ax,key,title in zip(axes,['macro_f1','prediction_mae_F','mia_auc_F'],
                                ['Test macro-F1','Prediction MAE to retrain on F','Forget MIA AUC (fixed loss attack)']):
            ax.bar(part['method'],part[key],color='#287da8')
            ax.set_title(title)
            ax.tick_params(axis='x',rotation=65)
            ax.grid(axis='y',alpha=.2)
            if key=='mia_auc_F': ax.axhline(.5,color='gray',linestyle='--',linewidth=1)
        fig.suptitle(f'ConPFU-inspired pilot, seed 42, forget fraction {fraction:.0%}; one seed, no confidence intervals')
        fig.savefig(output/f'fu_fraction_{fraction}.png',dpi=160)
        plt.close(fig)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    plot(args.run,args.output)
