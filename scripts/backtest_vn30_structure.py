from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.local_data import load_raw
from scripts.build_vn30_metrics import SYMBOLS

OUT=ROOT/'data'/'processed'

def future_return(series,h):
    return series.pct_change(h).shift(-h)*100

def main():
    metrics_path=OUT/'vn30_metrics_history.parquet'
    if not metrics_path.exists():
        raise FileNotFoundError('Run build_vn30_metrics.py first')
    m=pd.read_parquet(metrics_path).sort_index()
    idx=load_raw('VNINDEX').set_index('time')['close'].sort_index()
    df=m.join(pd.DataFrame({'ret_20d':future_return(idx,20),'ret_60d':future_return(idx,60),'vol_20d_fwd':idx.pct_change().rolling(20).std().shift(-20)*np.sqrt(252)*100}),how='inner')
    rows=[]
    for factor in ['dispersion_pct_252','top5_risk_share_pct_252']:
        for label,mask in [('low',df[factor]<=20),('normal',(df[factor]>20)&(df[factor]<80)),('high',df[factor]>=80)]:
            x=df.loc[mask]
            rows.append({'factor':factor,'bucket':label,'observations':len(x),'avg_ret_20d':x.ret_20d.mean(),'median_ret_20d':x.ret_20d.median(),'avg_ret_60d':x.ret_60d.mean(),'avg_fwd_vol':x.vol_20d_fwd.mean()})
    report=pd.DataFrame(rows)
    report.to_parquet(OUT/'vn30_structure_backtest.parquet',index=False)
    print(report.to_string(index=False))

if __name__=='__main__': main()
