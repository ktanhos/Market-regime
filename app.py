from datetime import date
import os
import time
import numpy as np
import pandas as pd
import streamlit as st
from vnstock import register_user
from vnstock.api.quote import Quote

st.set_page_config(page_title="Vietnam Market Regime", page_icon="📊", layout="wide")

VN30_FALLBACK = ["ACB","BCM","BID","BVH","CTG","FPT","GAS","GVR","HDB","HPG","MBB","MSN","MWG","PLX","SAB","SHB","SSB","SSI","STB","TCB","TPB","VCB","VHM","VIB","VIC","VJC","VNM","VPB","VRE","VIX"]
REQUEST_INTERVAL_SECONDS = 1.1
HISTORY_START = date(2015, 1, 1)

st.markdown("""<style>
.block-container{max-width:1280px;padding-top:2rem;padding-bottom:3rem}.hero{padding:1.8rem;border:1px solid rgba(49,51,63,.14);border-radius:18px;background:linear-gradient(135deg,rgba(18,52,86,.07),rgba(255,255,255,.02));margin-bottom:1rem}.hero h1{margin:0 0 .45rem;font-size:2rem}.hero p{margin:0;color:#667085;line-height:1.65;max-width:920px}.eyebrow{font-size:.78rem;font-weight:700;letter-spacing:.08em;color:#667085;text-transform:uppercase;margin-bottom:.55rem}.status-box{padding:1.4rem 1.5rem;border:1px solid rgba(49,51,63,.14);border-radius:16px;background:rgba(250,250,250,.45);margin-bottom:1rem}.status-title{font-size:.8rem;color:#667085;text-transform:uppercase;letter-spacing:.06em;margin-bottom:.35rem}.status-value{font-size:1.75rem;font-weight:700;margin-bottom:.35rem}.status-text,.section-note,.mini-note{color:#667085;line-height:1.55}.mini-card{padding:1rem 1.1rem;border:1px solid rgba(49,51,63,.12);border-radius:14px;min-height:125px}.mini-label{font-size:.82rem;color:#667085;margin-bottom:.45rem}.mini-value{font-size:1.15rem;font-weight:700;margin-bottom:.35rem}.section-title{font-size:1.25rem;font-weight:700;margin:1.35rem 0 .25rem}</style>""", unsafe_allow_html=True)

def clean(raw, symbol):
    if raw is None or raw.empty: raise ValueError(f"Không có dữ liệu cho {symbol}")
    df=raw.copy(); df.rename(columns={"date":"time","Date":"time","Time":"time"},inplace=True)
    required=["time","open","high","low","close","volume"]
    missing=[c for c in required if c not in df.columns]
    if missing: raise ValueError(f"Thiếu cột {missing} cho {symbol}")
    df=df[required].copy(); df["time"]=pd.to_datetime(df["time"],errors="coerce")
    for col in required[1:]: df[col]=pd.to_numeric(df[col],errors="coerce")
    return df.dropna(subset=["time","close"]).sort_values("time").drop_duplicates("time",keep="last")

def configure_vnstock(api_key):
    key=(api_key or os.getenv("VNSTOCK_API_KEY","")).strip()
    if not key: raise ValueError("Thiếu API key VNStock")
    os.environ["VNSTOCK_API_KEY"]=key; register_user(api_key=key)

def rate_limit_gate():
    previous=st.session_state.get("last_vnstock_request",0.0); wait=REQUEST_INTERVAL_SECONDS-(time.monotonic()-previous)
    if wait>0: time.sleep(wait)
    st.session_state["last_vnstock_request"]=time.monotonic()

def safe_error(exc):
    key=os.getenv("VNSTOCK_API_KEY",""); return str(exc).replace(key,"[API_KEY]")[:500]

def fetch_symbol(symbol,start,end):
    errors=[]
    for source in ["KBS","VCI"]:
        rate_limit_gate()
        try: return clean(Quote(symbol=symbol,source=source).history(start=start,end=end,interval="1D"),symbol)
        except Exception as exc: errors.append(f"{source}: {safe_error(exc)}")
    raise RuntimeError(f"Không lấy được {symbol}: "+" | ".join(errors))

def fetch_full_history(symbol,start_date,end_date):
    cursor=pd.Timestamp(start_date); end_ts=pd.Timestamp(end_date); pieces=[]
    while cursor<=end_ts:
        chunk_end=min(cursor+pd.Timedelta(days=119),end_ts)
        pieces.append(fetch_symbol(symbol,cursor.strftime("%Y-%m-%d"),chunk_end.strftime("%Y-%m-%d")))
        cursor=chunk_end+pd.Timedelta(days=1)
    return pd.concat(pieces,ignore_index=True).drop_duplicates("time",keep="last").sort_values("time").reset_index(drop=True)

def get_vn30_symbols():
    try:
        from vnstock import Reference
        listing=Reference().equity.list_by_group("VN30"); cols={str(c).lower():c for c in listing.columns}; col=cols.get("symbol") or cols.get("ticker") or cols.get("code")
        if col:
            symbols=sorted(listing[col].dropna().astype(str).str.upper().unique().tolist())
            if len(symbols)>=20: return symbols[:30],"Danh sách VN30 hiện tại"
    except Exception: pass
    return VN30_FALLBACK,"Danh sách VN30 dự phòng"

def calculate_features(df,level=49,park_window=22,stress_window=252):
    out=df.copy(); close=out["close"]
    out["strength"]=(close.pct_change(63)*.4+close.pct_change(126)*.2+close.pct_change(189)*.2+close.pct_change(252)*.2)*100
    out["roro_equal"]=out["strength"].rolling(level,min_periods=level).mean(); out["roro"]=out["strength"]-out["roro_equal"]
    out["trend_state"]=np.select([out["roro"]>0,out["roro"]<0],["TÍCH CỰC","SUY YẾU"],default="TRUNG TÍNH")
    log_range=np.log(out["high"]/out["low"])
    out["parkinson_vol"]=np.sqrt(log_range.pow(2).rolling(park_window,min_periods=park_window).mean()/(4*np.log(2))*252)*100
    base=out["parkinson_vol"].rolling(stress_window,min_periods=max(60,park_window)).mean(); std=out["parkinson_vol"].rolling(stress_window,min_periods=max(60,park_window)).std().replace(0,np.nan)
    out["stress_z"]=(out["parkinson_vol"]-base)/std
    out["stress_state"]=np.select([out["stress_z"]>=1.5,out["stress_z"]>=.5,out["stress_z"]<=-.5],["RẤT CAO","CAO","THẤP"],default="BÌNH THƯỜNG")
    return out

def build_recent_breadth(api_key,end_date):
    configure_vnstock(api_key); symbols,source=get_vn30_symbols(); start=(pd.Timestamp(end_date)-pd.DateOffset(days=450)).strftime("%Y-%m-%d"); end=pd.Timestamp(end_date).strftime("%Y-%m-%d")
    prices,failed={},[]; progress=st.progress(0,text="Đang kiểm tra sức khỏe của VN30"); status=st.empty()
    for i,symbol in enumerate(symbols,1):
        status.caption(f"Đang xử lý {i}/{len(symbols)} mã: {symbol}")
        try:
            df=fetch_symbol(symbol,start,end)
            if len(df)>=200: prices[symbol]=df.set_index("time")["close"]
            else: failed.append(f"{symbol}: chưa đủ dữ liệu")
        except Exception as exc: failed.append(f"{symbol}: {type(exc).__name__}")
        progress.progress(i/len(symbols))
    progress.empty(); status.empty()
    if len(prices)<20: raise RuntimeError("Không đủ dữ liệu để đánh giá sức khỏe VN30")
    panel=pd.DataFrame(prices).sort_index(); last=panel.iloc[-1]; prev=panel.iloc[-2]; ma20=panel.rolling(20,min_periods=20).mean().iloc[-1]; ma50=panel.rolling(50,min_periods=50).mean().iloc[-1]; ma200=panel.rolling(200,min_periods=200).mean().iloc[-1]
    def calc(ma):
        valid=last.notna()&ma.notna(); return (last[valid]>ma[valid]).mean()*100
    p20,p50,p200=calc(ma20),calc(ma50),calc(ma200); valid=last.notna()&prev.notna(); adv=(last[valid]>prev[valid]).mean()*100
    score=float(np.nanmean([p20,p50,p200,adv])); state="RẤT KHỎE" if score>=70 else "KHỎE" if score>=55 else "CÂN BẰNG" if score>=45 else "YẾU" if score>=30 else "RẤT YẾU"
    details=[]
    for s in panel.columns:
        details.append({"Mã":s,"Trên MA20":"Có" if pd.notna(ma20[s]) and last[s]>ma20[s] else "Không","Trên MA50":"Có" if pd.notna(ma50[s]) and last[s]>ma50[s] else "Không","Trên MA200":"Có" if pd.notna(ma200[s]) and last[s]>ma200[s] else "Không","Tăng phiên gần nhất":"Có" if pd.notna(prev[s]) and last[s]>prev[s] else "Không"})
    return {"date":panel.index.max(),"symbols":len(symbols),"valid_symbols":len(prices),"failed":failed,"source":source,"pct_above_ma20":p20,"pct_above_ma50":p50,"pct_above_ma200":p200,"pct_advancers":adv,"breadth_score":score,"breadth_state":state,"details":pd.DataFrame(details).sort_values("Mã")}

def market_state(trend,stress):
    if trend=="TÍCH CỰC" and stress=="THẤP": return "THỊ TRƯỜNG TÍCH CỰC","Xu hướng đang tốt và biến động thấp hơn mức thông thường.","Tốt"
    if trend=="TÍCH CỰC" and stress in ["CAO","RẤT CAO"]: return "TĂNG NHƯNG BIẾN ĐỘNG CAO","Xu hướng vẫn tích cực nhưng thị trường đang biến động mạnh hơn bình thường.","Cần theo dõi"
    if trend=="SUY YẾU" and stress in ["CAO","RẤT CAO"]: return "THỊ TRƯỜNG ĐANG CHỊU ÁP LỰC","Xu hướng suy yếu đi cùng mức biến động cao.","Rủi ro cao"
    if trend=="SUY YẾU" and stress=="THẤP": return "ÁP LỰC ĐANG HẠ NHIỆT","Xu hướng chưa tích cực nhưng mức biến động đã giảm.","Chuyển tiếp"
    return "GIAI ĐOẠN CHUYỂN TIẾP","Xu hướng và mức biến động chưa cùng cho một tín hiệu rõ ràng.","Cần quan sát"

def breadth_message(b):
    if b is None: return "Có thể cập nhật khi cần để xem nhóm cổ phiếu vốn hóa lớn đang đồng thuận đến đâu."
    if b["breadth_state"] in ["RẤT KHỎE","KHỎE"]: return "Nhiều cổ phiếu VN30 đang cùng duy trì trạng thái tích cực, giúp xác nhận thêm cho diễn biến chung."
    if b["breadth_state"] in ["YẾU","RẤT YẾU"]: return "Số cổ phiếu VN30 có trạng thái tích cực còn hạn chế, đây là dấu hiệu cần theo dõi thêm."
    return "Sức khỏe VN30 đang ở mức trung tính, chưa tạo thêm tín hiệu xác nhận rõ ràng."

def card(label,value,note): st.markdown(f"<div class='mini-card'><div class='mini-label'>{label}</div><div class='mini-value'>{value}</div><div class='mini-note'>{note}</div></div>",unsafe_allow_html=True)

st.markdown("""<div class='hero'><div class='eyebrow'>Theo dõi trạng thái thị trường</div><h1>Vietnam Market Regime</h1><p>Ứng dụng tập trung vào một câu hỏi: thị trường chứng khoán Việt Nam hiện đang vận động như thế nào. Kết quả được xây từ xu hướng và mức độ biến động của VNINDEX, sau đó dùng tình trạng hiện tại của nhóm VN30 để bổ sung góc nhìn về sức khỏe của các cổ phiếu vốn hóa lớn.</p></div>""",unsafe_allow_html=True)

with st.sidebar:
    st.header("Cập nhật dữ liệu"); key=st.text_input("API key VNStock",type="password",value=st.session_state.get("vnstock_api_key",""))
    if key: st.session_state["vnstock_api_key"]=key
    end=st.date_input("Dữ liệu đến ngày",date.today()); use_breadth=st.checkbox("Cập nhật sức khỏe VN30",value=True); run=st.button("Cập nhật trạng thái thị trường",type="primary",use_container_width=True)
    st.caption("VNINDEX được lấy từ năm 2015 để tạo nền so sánh. VN30 chỉ dùng để đánh giá tình trạng hiện tại của nhóm cổ phiếu lớn.")

api_key=st.session_state.get("vnstock_api_key","").strip()
if run:
    if not api_key: st.error("Vui lòng nhập API key VNStock."); st.stop()
    try:
        configure_vnstock(api_key)
        with st.status("Đang cập nhật dữ liệu",expanded=True) as status:
            st.write("Đang lấy VNINDEX và xây thước đo xu hướng, biến động"); vnindex=calculate_features(fetch_full_history("VNINDEX",HISTORY_START,end)); breadth=build_recent_breadth(api_key,end) if use_breadth else None; status.update(label="Đã hoàn tất",state="complete")
        st.session_state.update({"vnindex":vnindex,"breadth":breadth})
    except Exception as exc: st.error(f"Không thể hoàn tất: {type(exc).__name__}: {safe_error(exc)}"); st.stop()

vnindex=st.session_state.get("vnindex"); breadth=st.session_state.get("breadth")
if vnindex is None:
    st.info("Nhập API key ở thanh bên và chọn Cập nhật trạng thái thị trường để bắt đầu.")
    with st.expander("Ứng dụng hoạt động như thế nào?"):
        st.write("Bước 1: VNINDEX cho biết xu hướng chung đang mạnh lên hay suy yếu."); st.write("Bước 2: khoảng dao động giá của VNINDEX cho biết thị trường đang ổn định hay biến động mạnh."); st.write("Bước 3: 30 cổ phiếu VN30 hiện tại được kiểm tra để xem nhóm cổ phiếu vốn hóa lớn có đang cùng xác nhận diễn biến chung hay không."); st.write("Kết quả là mô tả trạng thái hiện tại, không phải dự báo giá trong vài phiên tới và không phải khuyến nghị mua bán.")
    st.stop()

latest=vnindex.dropna(subset=["roro","stress_state"]).iloc[-1]; regime,regime_note,regime_risk=market_state(latest["trend_state"],latest["stress_state"]); latest_date=pd.Timestamp(latest["time"]).strftime("%d/%m/%Y")
summary,market_tab,vn30_tab,guide_tab=st.tabs(["Tổng quan","Thị trường","VN30","Cách đọc kết quả"])
with summary:
    st.markdown(f"<div class='section-title'>Trạng thái mới nhất</div><div class='section-note'>Dựa trên dữ liệu VNINDEX đến ngày {latest_date}.</div>",unsafe_allow_html=True)
    st.markdown(f"<div class='status-box'><div class='status-title'>Đánh giá chung</div><div class='status-value'>{regime}</div><div class='status-text'>{regime_note}</div></div>",unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    with c1: card("Xu hướng VNINDEX",str(latest["trend_state"]).title(),f"RORO: {latest['roro']:.2f}")
    with c2: card("Mức biến động",str(latest["stress_state"]).title(),f"Biến động hiện tại: {latest['parkinson_vol']:.2f}%")
    with c3: card("Sức khỏe VN30",str(breadth["breadth_state"]).title() if breadth else "Chưa cập nhật",f"Điểm tổng hợp: {breadth['breadth_score']:.1f}/100" if breadth else "Có thể cập nhật riêng khi cần")
    with c4: card("Mức độ cần theo dõi",regime_risk,"Đánh giá từ xu hướng và biến động")
    st.markdown("<div class='section-title'>Ba góc nhìn tạo nên kết quả</div>",unsafe_allow_html=True)
    a,b,c=st.columns(3)
    with a: card("1. Xu hướng",str(latest["trend_state"]).title(),"VNINDEX đang mạnh lên hay suy yếu so với các giai đoạn trước.")
    with b: card("2. Biến động",str(latest["stress_state"]).title(),"Thị trường đang ổn định hay biến động mạnh hơn mức thường thấy.")
    with c: card("3. Sức khỏe VN30",str(breadth["breadth_state"]).title() if breadth else "Chưa cập nhật",breadth_message(breadth))
with market_tab:
    st.markdown("<div class='section-title'>Thị trường</div><div class='section-note'>VNINDEX là trục chính. Dữ liệu lịch sử được dùng để so sánh trạng thái hiện tại với chính thị trường trong quá khứ.</div>",unsafe_allow_html=True)
    left,right=st.columns(2)
    with left:
        st.subheader("Xu hướng"); st.metric("RORO hiện tại",f"{latest['roro']:.2f}",str(latest["trend_state"]).title()); st.line_chart(vnindex[["time","roro"]].dropna().tail(500).set_index("time"),height=280); st.caption("RORO dương cho thấy xu hướng hiện tại mạnh hơn mức trung bình gần đây. RORO âm cho thấy xu hướng đang suy yếu.")
    with right:
        st.subheader("Mức độ biến động"); st.metric("Biến động hiện tại",f"{latest['parkinson_vol']:.2f}%",str(latest["stress_state"]).title()); st.line_chart(vnindex[["time","parkinson_vol"]].dropna().tail(500).set_index("time"),height=280); st.caption("Mức biến động được so sánh với lịch sử gần đây để xác định thị trường đang ổn định, cao hay rất cao.")
    st.markdown("<div class='section-title'>Diễn giải hiện tại</div>",unsafe_allow_html=True); st.write(regime_note)
with vn30_tab:
    st.markdown("<div class='section-title'>Sức khỏe VN30 hiện tại</div><div class='section-note'>Phần này chỉ xem tình trạng hiện tại của nhóm cổ phiếu VN30, nhằm bổ sung góc nhìn về mức độ đồng thuận của các cổ phiếu vốn hóa lớn.</div>",unsafe_allow_html=True)
    if breadth is None: st.info("VN30 chưa được cập nhật trong lần chạy này. Chọn Cập nhật sức khỏe VN30 ở thanh bên và chạy lại khi cần.")
    else:
        b1,b2,b3,b4=st.columns(4)
        with b1: st.metric("Trên MA20",f"{breadth['pct_above_ma20']:.1f}%")
        with b2: st.metric("Trên MA50",f"{breadth['pct_above_ma50']:.1f}%")
        with b3: st.metric("Trên MA200",f"{breadth['pct_above_ma200']:.1f}%")
        with b4: st.metric("Tăng phiên gần nhất",f"{breadth['pct_advancers']:.1f}%")
        st.write(breadth_message(breadth)); st.caption(f"Dữ liệu hợp lệ: {breadth['valid_symbols']}/{breadth['symbols']} mã. Nguồn danh sách: {breadth['source']}."); st.dataframe(breadth["details"],use_container_width=True,hide_index=True)
        if breadth["failed"]:
            with st.expander("Các mã chưa lấy được dữ liệu"): st.write(" | ".join(breadth["failed"]))
with guide_tab:
    st.markdown("<div class='section-title'>Cách đọc kết quả</div><div class='section-note'>Ứng dụng không cố dự báo giá ngày mai. Mục tiêu là mô tả thị trường đang ở trạng thái nào tại thời điểm mới nhất.</div>",unsafe_allow_html=True)
    st.markdown("**Xu hướng** cho biết thị trường đang mạnh lên hay suy yếu. **Biến động** cho biết thị trường đang bình thường hay căng thẳng hơn mức thường thấy. **Sức khỏe VN30** cho biết nhóm cổ phiếu vốn hóa lớn có đang cùng xác nhận diễn biến chung hay không.")
    st.markdown("**Ví dụ:** xu hướng tích cực cộng biến động thấp cho thấy thị trường đang thuận lợi hơn. Xu hướng vẫn tích cực nhưng biến động cao cho thấy thị trường tăng nhưng rủi ro đã tăng lên. Xu hướng suy yếu đi cùng biến động cao cho thấy thị trường đang chịu áp lực.")
    st.info("Kết quả chỉ nhằm hỗ trợ quan sát và quản trị danh mục. Không phải khuyến nghị mua bán và không đảm bảo dự báo diễn biến giá trong tương lai.")