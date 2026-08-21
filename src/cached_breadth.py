from pathlib import Path
import numpy as np
import pandas as pd
from src.local_data import load_raw
from scripts.build_vn30_metrics import SYMBOLS

ROOT=Path(__file__).resolve().parents[1]

def build_cached_breadth():
    prices={}; failed=[]
    for symbol in SYMBOLS:
        df=load_raw(symbol)
        if df is None or len(df)<200:
            failed.append(symbol); continue
        prices[symbol]=df.set_index('time')['close']
    if len(prices)<20:
        raise RuntimeError('Không đủ dữ liệu VN30 đã lưu để đánh giá')
    panel=pd.DataFrame(prices).sort_index().ffill()
    last=panel.iloc[-1]; prev=panel.iloc[-2]
    ma20=panel.rolling(20,min_periods=20).mean().iloc[-1]
    ma50=panel.rolling(50,min_periods=50).mean().iloc[-1]
    ma200=panel.rolling(200,min_periods=200).mean().iloc[-1]
    def above(ma):
        valid=last.notna() & ma.notna()
        return float((last[valid]>ma[valid]).mean()*100)
    p20,p50,p200=above(ma20),above(ma50),above(ma200)
    valid=last.notna() & prev.notna(); adv=float((last[valid]>prev[valid]).mean()*100)
    score=float(np.nanmean([p20,p50,p200,adv]))
    state='RẤT KHỎE' if score>=70 else 'KHỎE' if score>=55 else 'CÂN BẰNG' if score>=45 else 'YẾU' if score>=30 else 'RẤT YẾU'
    returns=panel.pct_change()
    dispersion=float(returns.std(axis=1).iloc[-1]*100)
    hist=returns.std(axis=1).dropna().iloc[-252:]*100
    q=float((hist<=dispersion).mean()*100) if len(hist) else np.nan
    dispersion_state='PHÂN HÓA CAO' if q>=80 else 'PHÂN HÓA THẤP' if q<=20 else 'PHÂN HÓA BÌNH THƯỜNG'
    vol=returns.iloc[-20:].std()*np.sqrt(252); vol=vol.dropna()
    weights=vol/vol.sum(); hhi=float((weights**2).sum()); effective=float(1/hhi) if hhi>0 else np.nan
    top5=float(weights.nlargest(5).sum()*100)
    conc_q=np.nan
    metrics_path=ROOT/'data'/'processed'/'vn30_metrics_history.parquet'
    if metrics_path.exists():
        m=pd.read_parquet(metrics_path).sort_index()
        if 'top5_risk_share_pct_252' in m and len(m): conc_q=float(m.iloc[-1]['top5_risk_share_pct_252'])
    concentration_state='TẬP TRUNG CAO' if pd.notna(conc_q) and conc_q>=80 else 'TẬP TRUNG THẤP' if pd.notna(conc_q) and conc_q<=20 else 'TẬP TRUNG BÌNH THƯỜNG'
    details=pd.DataFrame({'Mã':panel.columns,'Trên MA20':['Có' if last[s]>ma20[s] else 'Không' for s in panel.columns],'Trên MA50':['Có' if last[s]>ma50[s] else 'Không' for s in panel.columns],'Trên MA200':['Có' if last[s]>ma200[s] else 'Không' for s in panel.columns],'Tăng phiên gần nhất':['Có' if last[s]>prev[s] else 'Không' for s in panel.columns]})
    risk_table=pd.DataFrame({'Mã':weights.index,'Biến động 20 phiên %':(vol*100).round(2),'Tỷ trọng biến động %':(weights*100).round(2)}).sort_values('Tỷ trọng biến động %',ascending=False)
    return {'date':panel.index.max(),'symbols':len(SYMBOLS),'valid_symbols':len(prices),'failed':failed,'source':'Dữ liệu nền đã lưu','pct_above_ma20':p20,'pct_above_ma50':p50,'pct_above_ma200':p200,'pct_advancers':adv,'breadth_score':score,'breadth_state':state,'dispersion':dispersion,'dispersion_percentile':q,'dispersion_state':dispersion_state,'effective_risk_names':effective,'top5_risk_share':top5,'concentration_percentile':conc_q,'concentration_state':concentration_state,'risk_table':risk_table,'details':details}
