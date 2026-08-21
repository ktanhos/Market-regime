from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data'/'raw'
PROCESSED=ROOT/'data'/'processed'

def load_latest():
    vnindex=pd.read_parquet(RAW/'vnindex.parquet')
    vnindex['time']=pd.to_datetime(vnindex['time'])
    metrics_path=PROCESSED/'vn30_metrics_history.parquet'
    metrics=pd.read_parquet(metrics_path) if metrics_path.exists() else None
    return vnindex.sort_values('time'), metrics

def freshness_status(symbols):
    rows=[]
    for s in symbols:
        p=RAW/f'{s.lower()}.parquet'
        if not p.exists():
            rows.append({'symbol':s,'status':'missing','last_date':pd.NaT})
        else:
            d=pd.read_parquet(p,columns=['time'])
            rows.append({'symbol':s,'status':'ok','last_date':pd.to_datetime(d['time']).max()})
    return pd.DataFrame(rows)
