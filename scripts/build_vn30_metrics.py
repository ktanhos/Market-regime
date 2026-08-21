from pathlib import Path
import numpy as np
import pandas as pd
from src.local_data import RAW_DIR, load_raw

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data'/'processed'
OUT.mkdir(parents=True,exist_ok=True)
SYMBOLS=['ACB','BCM','BID','BVH','CTG','FPT','GAS','GVR','HDB','HPG','MBB','MSN','MWG','PLX','SAB','SHB','SSB','SSI','STB','TCB','TPB','VCB','VHM','VIB','VIC','VJC','VNM','VPB','VRE','VIX']

def main():
    series={}
    for s in SYMBOLS:
        df=load_raw(s)
        if df is not None and 'close' in df and len(df)>30:
            series[s]=df.set_index('time')['close']
    panel=pd.DataFrame(series).sort_index()
    ret=panel.pct_change()
    dispersion=ret.std(axis=1)*100
    vol=ret.rolling(20).std()*np.sqrt(252)
    weights=vol.div(vol.sum(axis=1),axis=0)
    hhi=weights.pow(2).sum(axis=1)
    effective=1/hhi.replace(0,np.nan)
    top5=weights.apply(lambda x:x.nlargest(5).sum(),axis=1)*100
    out=pd.DataFrame({'dispersion':dispersion,'hhi':hhi,'effective_risk_names':effective,'top5_risk_share':top5})
    for c in ['dispersion','hhi','top5_risk_share']:
        out[c+'_pct_252']=out[c].rolling(252,min_periods=60).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1]*100,raw=False)
    out.to_parquet(OUT/'vn30_metrics_history.parquet')
    print(out.tail())

if __name__=='__main__': main()
