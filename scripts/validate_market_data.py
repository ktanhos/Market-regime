from pathlib import Path
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.local_data import load_raw
from scripts.build_vn30_metrics import SYMBOLS

OUT=ROOT/'data'/'processed'

def main():
    rows=[]
    for s in ['VNINDEX']+SYMBOLS:
        df=load_raw(s)
        if df is None or df.empty:
            rows.append({'symbol':s,'status':'missing','rows':0,'first_date':None,'last_date':None,'duplicates':0,'missing_close':0})
            continue
        t=pd.to_datetime(df['time'])
        rows.append({'symbol':s,'status':'ok','rows':len(df),'first_date':t.min(),'last_date':t.max(),'duplicates':int(t.duplicated().sum()),'missing_close':int(df['close'].isna().sum())})
    report=pd.DataFrame(rows)
    OUT.mkdir(parents=True,exist_ok=True)
    report.to_parquet(OUT/'data_quality_report.parquet',index=False)
    print(report.to_string(index=False))
    print('Latest-date spread:',report.loc[report.status=='ok','last_date'].max()-report.loc[report.status=='ok','last_date'].min())

if __name__=='__main__': main()
