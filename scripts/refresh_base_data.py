from datetime import date, timedelta
from pathlib import Path
import time
import pandas as pd
from vnstock import register_user
from vnstock.api.quote import Quote

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data' / 'raw'
DATA_DIR.mkdir(parents=True, exist_ok=True)
SYMBOLS = ['VNINDEX','ACB','BCM','BID','BVH','CTG','FPT','GAS','GVR','HDB','HPG','MBB','MSN','MWG','PLX','SAB','SHB','SSB','SSI','STB','TCB','TPB','VCB','VHM','VIB','VIC','VJC','VNM','VPB','VRE','VIX']
INTERVAL = 1.1

def clean(df):
    df=df.copy().rename(columns={'date':'time','Date':'time','Time':'time'})
    cols=['time','open','high','low','close','volume']
    df=df[[c for c in cols if c in df.columns]]
    df['time']=pd.to_datetime(df['time'],errors='coerce')
    return df.dropna(subset=['time','close']).sort_values('time').drop_duplicates('time',keep='last')

def fetch(symbol,start,end):
    errors=[]
    for source in ['KBS','VCI']:
        try:
            return clean(Quote(symbol=symbol,source=source).history(start=start,end=end,interval='1D'))
        except Exception as exc:
            errors.append(str(exc)[:120])
        time.sleep(INTERVAL)
    raise RuntimeError(symbol+': '+' | '.join(errors))

def update(symbol):
    path=DATA_DIR/f'{symbol.lower()}.parquet'
    end=date.today().isoformat()
    if path.exists():
        old=clean(pd.read_parquet(path)); last=old['time'].max().date()
        start=(last-timedelta(days=15)).isoformat()
        new=fetch(symbol,start,end)
        out=pd.concat([old,new],ignore_index=True)
    else:
        start='2015-01-01'; out=fetch(symbol,start,end)
    out=clean(out)
    out.to_parquet(path,index=False)
    print(symbol,len(out),out['time'].min().date(),out['time'].max().date())

def main():
    key=input('VNStock API key: ').strip()
    register_user(api_key=key)
    for symbol in SYMBOLS:
        try: update(symbol)
        except Exception as exc: print('FAILED',symbol,exc)
        time.sleep(INTERVAL)

if __name__=='__main__': main()
