from datetime import date
import queue
import threading
import time
import numpy as np
import pandas as pd
import streamlit as st
from vnstock import Market, Reference, register_user

st.set_page_config(page_title="Vietnam Market Regime", page_icon="📊", layout="wide")

VN30_FALLBACK=["ACB","BCM","BID","BVH","CTG","FPT","GAS","GVR","HDB","HPG","MBB","MSN","MWG","PLX","SAB","SHB","SSB","SSI","STB","TCB","TPB","VCB","VHM","VIB","VIC","VJC","VNM","VPB","VRE","VIX"]

def clean(df):
    if df is None or df.empty: raise ValueError("Dữ liệu rỗng")
    x=df.copy(); x.columns=[str(c).lower() for c in x.columns]
    dc="time" if "time" in x else "date"
    x=x.rename(columns={dc:"date"}); x["date"]=pd.to_datetime(x["date"]).dt.normalize()
    for c in ["open","high","low","close"]:
        x[c]=pd.to_numeric(x[c],errors="coerce")
    return x.dropna(subset=["date","close"]).drop_duplicates("date").sort_values("date")

def timeout_call(fn,**kw):
    q=queue.Queue(1)
    def run():
        try:q.put((1,fn(**kw)))
        except Exception as e:q.put((0,e))
    threading.Thread(target=run,daemon=True).start()
    try:ok,v=q.get(timeout=40)
    except queue.Empty:raise TimeoutError("Quá 40 giây")
    if not ok:raise v
    return v

def history(fn,start,end,retries=2):
    parts=[]; errors=[]; a=pd.Timestamp(start); end=pd.Timestamp(end)
    while a<=end:
        b=min(a+pd.Timedelta(days=119),end); last=None
        for k in range(retries):
            try:
                parts.append(clean(timeout_call(fn,start=a.strftime("%Y-%m-%d"),end=b.strftime("%Y-%m-%d"),interval="1D"))); last=None; break
            except Exception as e:
                last=e; time.sleep(1+k)
        if last: errors.append(type(last).__name__)
        a=b+pd.Timedelta(days=1)
    if not parts: raise ConnectionError("Không lấy được dữ liệu")
    return pd.concat(parts).drop_duplicates("date").sort_values("date"),errors

def features(df):
    x=df.copy(); c=x.close
    x["strength"]=(c.pct_change(63)*.4+c.pct_change(126)*.2+c.pct_change(189)*.2+c.pct_change(252)*.2)*100
    x["roro"]=x.strength-x.strength.rolling(49,min_periods=49).mean()
    lr=np.log(x.high/x.low); x["parkinson_vol"]=np.sqrt(lr.pow(2).rolling(22,min_periods=22).mean()/(4*np.log(2))*252)*100
    m=x.parkinson_vol.rolling(252,min_periods=60).mean(); s=x.parkinson_vol.rolling(252,min_periods=60).std().replace(0,np.nan)
    x["stress_z"]=(x.parkinson_vol-m)/s; x["stress_score"]=50+10*x.stress_z.clip(-5,5)
    x["regime"]=np.select([x.roro>0,x.stress_z>=1.5,x.stress_z<.5],["EXPANSION","HIGH STRESS RISK OFF","CORRECTION"],default="TRANSITION")
    x.loc[(x.roro>0)&(x.stress_z>=.5),"regime"]="FRAGILE RALLY"
    return x

def validation(x):
    z=x[["regime","close"]].copy(); rows=[]
    for h in [5,10,20]:z[f"f{h}"]=(z.close.shift(-h)/z.close-1)*100
    for r,g in z.groupby("regime"):
        if r=="CHƯA ĐỦ DỮ LIỆU":continue
        row={"Regime":r,"Số quan sát":len(g)}
        for h in [5,10,20]:
            a=g[f"f{h}"].dropna(); row[f"T{h} lợi suất TB %"]=a.mean(); row[f"Tỷ lệ dương T{h} %"]=(a>0).mean()*100
        rows.append(row)
    return pd.DataFrame(rows)

def symbols():
    try:
        d=Reference().equity.list_by_group("VN30"); col=next(c for c in d.columns if str(c).lower() in ["symbol","ticker","code"])
        s=d[col].dropna().astype(str).str.upper().unique().tolist()
        if len(s)>=20:return s[:30]
    except Exception:pass
    return VN30_FALLBACK

def current_breadth(market,end):
    syms=symbols(); series={}; failed=[]; p=st.progress(0,text="Đang lấy Breadth VN30 hiện tại")
    start=pd.Timestamp(end)-pd.Timedelta(days=430)
    for i,sym in enumerate(syms,1):
        try:
            d,_=history(market.equity(sym).ohlcv,start,end,retries=1)
            if len(d)>=200:series[sym]=d.set_index("date").close
            else:failed.append(sym)
        except Exception:failed.append(sym)
        p.progress(i/30,text=f"Breadth hiện tại {i}/30: {sym}")
    p.empty()
    px=pd.DataFrame(series).sort_index(); px=px.tail(200); last=px.iloc[-1]; valid=last.notna(); n=int(valid.sum())
    ma20=px.rolling(20,min_periods=20).mean().iloc[-1]; ma50=px.rolling(50,min_periods=50).mean().iloc[-1]; ma200=px.rolling(200,min_periods=200).mean().iloc[-1]
    adv=(pd.DataFrame(series).pct_change().iloc[-1][valid]>0).mean()*100
    vals=[(last[valid]>ma20[valid]).mean()*100,(last[valid]>ma50[valid]).mean()*100,(last[valid]>ma200[valid]).mean()*100,adv]
    score=float(np.nanmean(vals)); state="RẤT RỘNG" if score>=70 else "RỘNG" if score>=55 else "TRUNG TÍNH" if score>=45 else "HẸP" if score>=30 else "RẤT HẸP"
    return {"n":n,"failed":failed,"ma20":vals[0],"ma50":vals[1],"ma200":vals[2],"adv":adv,"score":score,"state":state,"date":px.index[-1]}

st.title("Vietnam Market Regime")
st.caption("Trend và Stress được kiểm định lịch sử. Breadth chỉ đánh giá trạng thái hiện tại của VN30.")
with st.sidebar:
    key=st.text_input("API key VNStock",type="password"); start=st.date_input("Từ ngày",date(2015,1,1)); end=st.date_input("Đến ngày",date.today()); use_b=st.checkbox("Tính Breadth VN30 hiện tại",True); run=st.button("Lấy dữ liệu và phân tích",type="primary")
if run:
    try:
        register_user(api_key=key.strip()); m=Market(); res={}
        for sym in ["VNINDEX","VN30"]:res[sym]=features(history(m.index(sym).ohlcv,start,end)[0])
        st.session_state["res"]=res; st.session_state["breadth"]=current_breadth(m,end) if use_b else None
    except Exception as e:st.error(f"Lỗi: {type(e).__name__}: {e}")
if "res" in st.session_state:
    for name,x in st.session_state.res.items():
        st.header(name); a=x.iloc[-1]; c=st.columns(4); c[0].metric("Đóng cửa",f"{a.close:,.2f}"); c[1].metric("RORO",f"{a.roro:.2f}"); c[2].metric("Stress",f"{a.stress_score:.1f}"); c[3].metric("Regime",a.regime)
        t=st.tabs(["Tổng quan","Trend","Stress","Kiểm định"])
        with t[0]:st.line_chart(x.set_index("date")[["close"]])
        with t[1]:st.line_chart(x.set_index("date")[["roro"]])
        with t[2]:st.line_chart(x.set_index("date")[["parkinson_vol","stress_score"]].dropna(how="all"))
        with t[3]:st.dataframe(validation(x).round(4),hide_index=True,use_container_width=True)
    b=st.session_state.get("breadth")
    if b:
        st.divider(); st.header("Breadth VN30 hiện tại"); st.caption("Ảnh chụp của 30 cổ phiếu VN30 hiện tại, không dùng để mô phỏng lịch sử từ năm 2015.")
        q=st.columns(5); q[0].metric("Trên MA20",f"{b['ma20']:.1f}%"); q[1].metric("Trên MA50",f"{b['ma50']:.1f}%"); q[2].metric("Trên MA200",f"{b['ma200']:.1f}%"); q[3].metric("Số mã tăng",f"{b['adv']:.1f}%"); q[4].metric("Breadth",f"{b['score']:.1f} | {b['state']}")
        st.info(f"Dữ liệu ngày {b['date'].date()}: {b['n']}/30 mã có dữ liệu. Không lấy được: {', '.join(b['failed']) if b['failed'] else 'Không có'}")
    st.divider(); st.header("Lộ trình mô hình"); st.write("Trend và Stress được kiểm định lịch sử từ năm 2015. Breadth hiện tại chỉ là lớp xác nhận trạng thái. Breadth lịch sử sẽ chỉ được xây khi có thành phần VN30 theo từng kỳ rà soát hoặc một vũ trụ thị trường lịch sử phù hợp.")
else:st.info("Nhập API key, chọn khoảng thời gian và chạy phân tích.")
