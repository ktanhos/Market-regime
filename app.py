from datetime import date
import os
import time
import numpy as np
import pandas as pd
import streamlit as st
from vnstock import register_user
from vnstock.api.quote import Quote

st.set_page_config(page_title="Trạng thái thị trường Việt Nam", page_icon="📊", layout="wide")

VN30_FALLBACK = ["ACB","BCM","BID","BVH","CTG","FPT","GAS","GVR","HDB","HPG","MBB","MSN","MWG","PLX","SAB","SHB","SSB","SSI","STB","TCB","TPB","VCB","VHM","VIB","VIC","VJC","VNM","VPB","VRE","VIX"]
REQUEST_INTERVAL_SECONDS = 1.1
HISTORY_START = date(2015, 1, 1)

st.markdown("""<style>
:root{--good:#15803d;--good-bg:#ecfdf3;--bad:#dc2626;--bad-bg:#fef2f2;--warn:#ca8a04;--warn-bg:#fefce8;--ink:#172033;--muted:#667085;--line:#e4e7ec}
.block-container{max-width:1280px;padding-top:1.5rem;padding-bottom:3rem}
.hero{padding:1.7rem 1.8rem;border:1px solid var(--line);border-radius:18px;background:#fff;margin-bottom:1rem;box-shadow:0 1px 3px rgba(16,24,40,.04)}
.hero h1{margin:0 0 .4rem;font-size:2rem;color:var(--ink)}.hero p{margin:0;color:var(--muted);line-height:1.65;max-width:900px}
.eyebrow{font-size:.76rem;font-weight:700;letter-spacing:.08em;color:var(--muted);text-transform:uppercase;margin-bottom:.5rem}
.status-box,.mini-card,.action-card{padding:1.2rem 1.25rem;border:1px solid var(--line);border-radius:15px;background:#fff;min-height:118px}
.status-box{border-left:5px solid var(--accent,#ca8a04);margin-bottom:1rem}.status-title,.mini-label{font-size:.8rem;color:var(--muted);margin-bottom:.4rem}.status-value{font-size:1.65rem;font-weight:750;color:var(--ink);margin-bottom:.35rem}.mini-value{font-size:1.25rem;font-weight:700;color:var(--ink);margin-bottom:.35rem}.status-text,.mini-note,.section-note{color:var(--muted);line-height:1.55}
.good{--accent:var(--good)}.bad{--accent:var(--bad)}.warn{--accent:var(--warn)}.neutral{--accent:#64748b}.good .mini-value,.good .status-value{color:var(--good)}.bad .mini-value,.bad .status-value{color:var(--bad)}.warn .mini-value,.warn .status-value{color:var(--warn)}
.section-title{font-size:1.25rem;font-weight:700;color:var(--ink);margin:1.5rem 0 .25rem}.badge{display:inline-block;padding:.28rem .65rem;border-radius:999px;font-size:.76rem;font-weight:700}.badge-good{color:var(--good);background:var(--good-bg)}.badge-bad{color:var(--bad);background:var(--bad-bg)}.badge-warn{color:#854d0e;background:var(--warn-bg)}.badge-neutral{color:#475467;background:#f2f4f7}
.action-card{border-left:5px solid var(--accent,#ca8a04);margin-top:.7rem}.action-card h3{margin:0 0 .4rem;color:var(--ink)}
</style>""", unsafe_allow_html=True)

def tone(text):
    x=str(text).upper()
    if any(k in x for k in ["TÍCH CỰC","KHỎE","THẤP","HẠ NHIỆT","PHÂN TÁN","ỔN ĐỊNH"]): return "good"
    if any(k in x for k in ["SUY YẾU","YẾU","RỦI RO","ÁP LỰC","CAO","TẬP TRUNG"]): return "bad"
    if any(k in x for k in ["CHUYỂN TIẾP","CÂN BẰNG","BÌNH THƯỜNG","CẦN"]): return "warn"
    return "neutral"

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
    prices,failed={},[]; progress=st.progress(0,text="Đang đánh giá VN30"); status=st.empty()
    for i,symbol in enumerate(symbols,1):
        status.caption(f"Đang xử lý {i}/{len(symbols)} mã: {symbol}")
        try:
            df=fetch_symbol(symbol,start,end)
            if len(df)>=200: prices[symbol]=df.set_index("time")["close"]
            else: failed.append(f"{symbol}: chưa đủ dữ liệu")
        except Exception as exc: failed.append(f"{symbol}: {type(exc).__name__}")
        progress.progress(i/len(symbols))
    progress.empty(); status.empty()
    if len(prices)<20: raise RuntimeError("Không đủ dữ liệu để đánh giá VN30")
    panel=pd.DataFrame(prices).sort_index(); last=panel.iloc[-1]; prev=panel.iloc[-2]
    ma20=panel.rolling(20,min_periods=20).mean().iloc[-1]; ma50=panel.rolling(50,min_periods=50).mean().iloc[-1]; ma200=panel.rolling(200,min_periods=200).mean().iloc[-1]
    def calc(ma):
        valid=last.notna()&ma.notna(); return (last[valid]>ma[valid]).mean()*100
    p20,p50,p200=calc(ma20),calc(ma50),calc(ma200); valid=last.notna()&prev.notna(); adv=(last[valid]>prev[valid]).mean()*100
    score=float(np.nanmean([p20,p50,p200,adv])); state="RẤT KHỎE" if score>=70 else "KHỎE" if score>=55 else "CÂN BẰNG" if score>=45 else "YẾU" if score>=30 else "RẤT YẾU"
    returns=panel.pct_change()
    cs_std=returns.std(axis=1,skipna=True)*100
    latest_disp=float(cs_std.iloc[-1]); hist_disp=cs_std.dropna().iloc[-252:]; q=float((hist_disp<=latest_disp).mean()*100) if len(hist_disp) else np.nan
    dispersion_state="PHÂN HÓA CAO" if q>=75 else "PHÂN HÓA THẤP" if q<=25 else "PHÂN HÓA TRUNG BÌNH"
    vol=returns.iloc[-20:].std()*np.sqrt(252); vol=vol.replace([np.inf,-np.inf],np.nan).dropna()
    weights=vol/vol.sum() if vol.sum()>0 else vol*0; hhi=float((weights**2).sum()); effective=float(1/hhi) if hhi>0 else np.nan
    top5_share=float(weights.sort_values(ascending=False).head(5).sum()*100)
    concentration_state="TẬP TRUNG CAO" if effective<15 else "TẬP TRUNG THẤP" if effective>22 else "TẬP TRUNG TRUNG BÌNH"
    risk_table=pd.DataFrame({"Mã":vol.index,"Biến động 20 phiên %":(vol*100).round(2),"Tỷ trọng biến động %":(weights*100).round(2)}).sort_values("Tỷ trọng biến động %",ascending=False)
    details=[]
    for s in panel.columns:
        details.append({"Mã":s,"Trên MA20":"Có" if pd.notna(ma20[s]) and last[s]>ma20[s] else "Không","Trên MA50":"Có" if pd.notna(ma50[s]) and last[s]>ma50[s] else "Không","Trên MA200":"Có" if pd.notna(ma200[s]) and last[s]>ma200[s] else "Không","Tăng phiên gần nhất":"Có" if pd.notna(prev[s]) and last[s]>prev[s] else "Không"})
    return {"date":panel.index.max(),"symbols":len(symbols),"valid_symbols":len(prices),"failed":failed,"source":source,"pct_above_ma20":p20,"pct_above_ma50":p50,"pct_above_ma200":p200,"pct_advancers":adv,"breadth_score":score,"breadth_state":state,"dispersion":latest_disp,"dispersion_percentile":q,"dispersion_state":dispersion_state,"effective_risk_names":effective,"top5_risk_share":top5_share,"concentration_state":concentration_state,"risk_table":risk_table,"details":pd.DataFrame(details).sort_values("Mã")}

def market_state(trend,stress):
    if trend=="TÍCH CỰC" and stress=="THẤP": return "THỊ TRƯỜNG TÍCH CỰC","Xu hướng đang tốt và mức biến động thấp hơn thông thường.","Tốt"
    if trend=="TÍCH CỰC" and stress in ["CAO","RẤT CAO"]: return "TĂNG NHƯNG BIẾN ĐỘNG CAO","Xu hướng vẫn tích cực nhưng rủi ro thị trường đã tăng.","Cần theo dõi"
    if trend=="SUY YẾU" and stress in ["CAO","RẤT CAO"]: return "THỊ TRƯỜNG ĐANG CHỊU ÁP LỰC","Xu hướng suy yếu đi cùng mức biến động cao.","Rủi ro cao"
    if trend=="SUY YẾU" and stress=="THẤP": return "ÁP LỰC ĐANG HẠ NHIỆT","Xu hướng chưa tích cực nhưng mức biến động đã giảm.","Chuyển tiếp"
    return "GIAI ĐOẠN CHUYỂN TIẾP","Xu hướng đang suy yếu hoặc chưa rõ, trong khi mức biến động chưa đủ cao để xác nhận một giai đoạn căng thẳng. Các tín hiệu hiện chưa cùng chỉ về một trạng thái thống nhất.","Cần quan sát"

def breadth_message(b):
    if b is None: return "Chưa cập nhật VN30. Có thể cập nhật khi cần để bổ sung góc nhìn về nhóm cổ phiếu vốn hóa lớn."
    health="Nhiều cổ phiếu đang cùng duy trì trạng thái tích cực." if b["breadth_state"] in ["RẤT KHỎE","KHỎE"] else "Số cổ phiếu có trạng thái tích cực còn hạn chế." if b["breadth_state"] in ["YẾU","RẤT YẾU"] else "Tín hiệu giữa các cổ phiếu đang cân bằng, chưa có sự đồng thuận rõ ràng."
    return health

def portfolio_guidance(regime,b):
    if regime=="THỊ TRƯỜNG TÍCH CỰC": return "CÓ THỂ TĂNG RỦI RO","Môi trường chung thuận lợi hơn. Có thể duy trì hoặc tăng dần mức đầu tư cổ phiếu nếu danh mục và khẩu vị rủi ro phù hợp. Không nên xem đây là lý do để bỏ qua kiểm soát tập trung và đòn bẩy.","good"
    if regime=="THỊ TRƯỜNG ĐANG CHỊU ÁP LỰC": return "ƯU TIÊN PHÒNG THỦ","Xu hướng yếu và biến động cao cùng xuất hiện. Nên giảm các vị thế rủi ro cao, hạn chế mở rộng đòn bẩy và ưu tiên thanh khoản, chất lượng danh mục.","bad"
    if regime=="TĂNG NHƯNG BIẾN ĐỘNG CAO": return "GIỮ RỦI RO Ở MỨC KIỂM SOÁT","Xu hướng chưa xấu nhưng biến động đã tăng. Phù hợp hơn với việc kiểm soát quy mô vị thế và tránh tăng rủi ro quá nhanh.","warn"
    if regime=="ÁP LỰC ĐANG HẠ NHIỆT": return "CÓ THỂ QUAN SÁT SỰ ỔN ĐỊNH","Biến động đã giảm nhưng xu hướng chưa xác nhận tích cực. Có thể đánh giá lại danh mục và chỉ tăng rủi ro khi xuất hiện thêm tín hiệu cải thiện.","warn"
    return "DUY TRÌ CÁCH TIẾP CẬN THẬN TRỌNG","Thị trường chưa có trạng thái thống nhất. Không nên thay đổi mạnh tổng mức rủi ro chỉ dựa trên một tín hiệu. Ưu tiên quản trị tỷ trọng, thanh khoản và chờ thêm sự xác nhận.","warn"

def card(label,value,note,style="neutral"):
    st.markdown(f"<div class='mini-card {style}'><div class='mini-label'>{label}</div><div class='mini-value'>{value}</div><div class='mini-note'>{note}</div></div>",unsafe_allow_html=True)

def badge(text):
    t=tone(text); return f"<span class='badge badge-{t}'>{text}</span>"

st.markdown("""<div class='hero'><div class='eyebrow'>Theo dõi trạng thái thị trường</div><h1>Vietnam Market Regime</h1><p>Ứng dụng giúp trả lời hai câu hỏi: thị trường hiện đang ở trạng thái nào và trong bối cảnh đó danh mục nên được quản trị với mức độ rủi ro ra sao. Xu hướng và biến động được đo từ VNINDEX. Nhóm VN30 được dùng để bổ sung góc nhìn về mức độ đồng thuận, phân hóa và sự tập trung biến động giữa các cổ phiếu vốn hóa lớn.</p></div>""",unsafe_allow_html=True)

with st.sidebar:
    st.header("Cập nhật dữ liệu")
    key=st.text_input("API key VNStock",type="password",value=st.session_state.get("vnstock_api_key",""))
    if key: st.session_state["vnstock_api_key"]=key
    end=st.date_input("Dữ liệu đến ngày",date.today())
    use_breadth=st.checkbox("Cập nhật VN30",value=True)
    run=st.button("Cập nhật trạng thái thị trường",type="primary",use_container_width=True)
    st.caption("VNINDEX được lấy từ năm 2015 để tạo nền so sánh. VN30 chỉ được dùng để đánh giá trạng thái hiện tại của nhóm cổ phiếu lớn.")

api_key=st.session_state.get("vnstock_api_key","").strip()
if run:
    if not api_key: st.error("Vui lòng nhập API key VNStock."); st.stop()
    try:
        configure_vnstock(api_key)
        with st.status("Đang cập nhật dữ liệu",expanded=True) as status:
            st.write("Đang lấy VNINDEX và xây thước đo xu hướng, biến động")
            vnindex=calculate_features(fetch_full_history("VNINDEX",HISTORY_START,end))
            breadth=build_recent_breadth(api_key,end) if use_breadth else None
            status.update(label="Đã hoàn tất",state="complete")
        st.session_state.update({"vnindex":vnindex,"breadth":breadth})
    except Exception as exc: st.error(f"Không thể hoàn tất: {type(exc).__name__}: {safe_error(exc)}"); st.stop()

vnindex=st.session_state.get("vnindex"); breadth=st.session_state.get("breadth")
if vnindex is None:
    st.info("Nhập API key ở thanh bên và chọn Cập nhật trạng thái thị trường để bắt đầu.")
    st.stop()

latest=vnindex.dropna(subset=["roro","parkinson_vol","stress_state"]).iloc[-1]
trend,stress=latest["trend_state"],latest["stress_state"]
regime,regime_text,attention=market_state(trend,stress)
allocation_title,allocation_text,allocation_style=portfolio_guidance(regime,breadth)

main,market_tab,vn30_tab,portfolio_tab,guide_tab=st.tabs(["Tổng quan","Thị trường","VN30","Quản trị danh mục","Cách đọc"])

with main:
    st.markdown(f"<div class='status-box {tone(regime)}'><div class='status-title'>Trạng thái thị trường hiện tại</div><div class='status-value'>{regime}</div><div class='status-text'>Dựa trên dữ liệu VNINDEX đến ngày {pd.Timestamp(latest['time']).strftime('%d/%m/%Y')}. {regime_text}</div></div>",unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    with c1: card("Xu hướng VNINDEX",trend,f"RORO: {latest['roro']:.2f}",tone(trend))
    with c2: card("Mức biến động",stress,f"Biến động hiện tại: {latest['parkinson_vol']:.2f}%",tone(stress))
    with c3:
        if breadth: card("Sức khỏe VN30",breadth['breadth_state'],f"Điểm tổng hợp: {breadth['breadth_score']:.1f}/100",tone(breadth['breadth_state']))
        else: card("Sức khỏe VN30","CHƯA CẬP NHẬT","Có thể cập nhật riêng khi cần",'neutral')
    with c4: card("Mức độ cần theo dõi",attention,"Đánh giá từ xu hướng và biến động",tone(attention))
    st.markdown("<div class='section-title'>Điều này có ý nghĩa gì với danh mục?</div>",unsafe_allow_html=True)
    st.markdown(f"<div class='action-card {allocation_style}'><h3>{allocation_title}</h3><div class='section-note'>{allocation_text}</div></div>",unsafe_allow_html=True)
    if breadth:
        st.markdown("<div class='section-title'>Trạng thái bên trong VN30</div>",unsafe_allow_html=True)
        a,b,c=st.columns(3)
        with a: card("Mức độ đồng thuận",breadth['breadth_state'],breadth_message(breadth),tone(breadth['breadth_state']))
        with b: card("Mức độ phân hóa",breadth['dispersion_state'],f"Độ phân tán lợi suất hiện tại ở phân vị {breadth['dispersion_percentile']:.0f} so với 252 phiên gần đây",tone(breadth['dispersion_state']))
        with c: card("Tập trung biến động",breadth['concentration_state'],f"5 mã biến động mạnh nhất chiếm {breadth['top5_risk_share']:.1f}% tổng biến động tương đối",tone(breadth['concentration_state']))

with market_tab:
    st.markdown("<div class='section-title'>Xu hướng và biến động</div><div class='section-note'>VNINDEX là dữ liệu chính để xác định trạng thái thị trường. Dữ liệu lịch sử được dùng để tạo thước đo và so sánh, không nhằm hiển thị lại biểu đồ giá đơn thuần.</div>",unsafe_allow_html=True)
    x1,x2=st.columns(2)
    with x1: st.line_chart(vnindex.set_index("time")[["roro"]].tail(500)); st.caption("RORO cho biết sức mạnh xu hướng hiện tại đang tốt lên hay suy yếu so với chính lịch sử gần đây.")
    with x2: st.line_chart(vnindex.set_index("time")[["parkinson_vol"]].tail(500)); st.caption("Biến động được so sánh với nền lịch sử để xác định thị trường đang bình thường hay căng thẳng hơn mức thường thấy.")

with vn30_tab:
    if not breadth:
        st.info("Chưa cập nhật dữ liệu VN30. Chọn cập nhật VN30 ở thanh bên để xem trạng thái hiện tại.")
    else:
        st.markdown("<div class='section-title'>Sức khỏe và cấu trúc VN30 hiện tại</div><div class='section-note'>Phần này chỉ đánh giá 30 cổ phiếu VN30 hiện tại. Mục tiêu là xem diễn biến có được xác nhận rộng rãi hay đang tập trung vào một nhóm nhỏ.</div>",unsafe_allow_html=True)
        c1,c2,c3,c4=st.columns(4)
        with c1: card("Trên MA20",f"{breadth['pct_above_ma20']:.1f}%","Xu hướng ngắn hạn",'good' if breadth['pct_above_ma20']>=60 else 'bad' if breadth['pct_above_ma20']<40 else 'warn')
        with c2: card("Trên MA50",f"{breadth['pct_above_ma50']:.1f}%","Xu hướng trung hạn",'good' if breadth['pct_above_ma50']>=60 else 'bad' if breadth['pct_above_ma50']<40 else 'warn')
        with c3: card("Trên MA200",f"{breadth['pct_above_ma200']:.1f}%","Xu hướng dài hạn",'good' if breadth['pct_above_ma200']>=60 else 'bad' if breadth['pct_above_ma200']<40 else 'warn')
        with c4: card("Tăng phiên gần nhất",f"{breadth['pct_advancers']:.1f}%","Sức lan tỏa trong phiên",'good' if breadth['pct_advancers']>=60 else 'bad' if breadth['pct_advancers']<40 else 'warn')
        st.markdown("<div class='section-title'>Phân hóa và tập trung biến động</div>",unsafe_allow_html=True)
        x,y=st.columns(2)
        with x: card("Mức độ phân hóa",breadth['dispersion_state'],f"Lợi suất giữa các cổ phiếu hiện ở phân vị {breadth['dispersion_percentile']:.0f} so với 252 phiên gần đây. Phân hóa cao nghĩa là các mã đang diễn biến khác nhau rõ hơn.",tone(breadth['dispersion_state']))
        with y: card("Tập trung biến động",breadth['concentration_state'],f"Số mã biến động tương đương: {breadth['effective_risk_names']:.1f}. 5 mã có biến động cao nhất chiếm {breadth['top5_risk_share']:.1f}% tổng biến động tương đối.",tone(breadth['concentration_state']))
        st.caption("Đây là thước đo mức độ tập trung biến động trong VN30, không phải mức đóng góp rủi ro chính xác của danh mục nhà đầu tư vì ứng dụng chưa biết tỷ trọng và tương quan trong danh mục thực tế.")
        st.markdown("<div class='section-title'>Những mã có biến động đóng góp tương đối lớn</div>",unsafe_allow_html=True)
        st.dataframe(breadth['risk_table'].head(10),use_container_width=True,hide_index=True)
        with st.expander("Xem chi tiết từng cổ phiếu"):
            st.dataframe(breadth['details'],use_container_width=True,hide_index=True)
        st.caption(f"Dữ liệu hợp lệ: {breadth['valid_symbols']}/{breadth['symbols']} mã. Nguồn danh sách: {breadth['source']}.")

with portfolio_tab:
    st.markdown("<div class='section-title'>Khung quản trị danh mục</div><div class='section-note'>Ứng dụng không chọn cổ phiếu thay người dùng. Phần này chuyển trạng thái thị trường thành mức độ rủi ro tham khảo để hỗ trợ quyết định về tổng tỷ trọng cổ phiếu, quy mô vị thế và đòn bẩy.</div>",unsafe_allow_html=True)
    st.markdown(f"<div class='action-card {allocation_style}'><h3>{allocation_title}</h3><div class='section-note'>{allocation_text}</div></div>",unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Các điểm cần xem xét</div>",unsafe_allow_html=True)
    if regime in ["THỊ TRƯỜNG ĐANG CHỊU ÁP LỰC","GIAI ĐOẠN CHUYỂN TIẾP"]:
        st.write("• Không tăng mạnh tổng mức rủi ro khi tín hiệu chưa được xác nhận")
        st.write("• Hạn chế đòn bẩy và các vị thế có biến động cao")
        st.write("• Kiểm tra mức tập trung của danh mục trước khi mở vị thế mới")
    else:
        st.write("• Điều chỉnh mức rủi ro theo xu hướng và biến động thay vì phản ứng với từng phiên")
        st.write("• Kiểm soát quy mô vị thế và mức tập trung vào từng cổ phiếu")
        st.write("• Nếu VN30 phân hóa cao, cần đánh giá từng cổ phiếu thay vì chỉ dựa vào chỉ số chung")
    if breadth and breadth['concentration_state']=="TẬP TRUNG CAO": st.warning("Biến động của VN30 đang tập trung vào một số mã. Cần đặc biệt kiểm tra mức độ tập trung nếu danh mục có các cổ phiếu này.")

with guide_tab:
    st.markdown("<div class='section-title'>Cách đọc kết quả</div>",unsafe_allow_html=True)
    st.write("Ứng dụng không cố dự đoán VNINDEX sẽ tăng hay giảm trong phiên tiếp theo. Mục tiêu là nhận biết thị trường đang ở trạng thái nào và trong bối cảnh đó mức rủi ro của danh mục nên được quản trị như thế nào.")
    st.write("• Xu hướng cho biết VNINDEX đang mạnh lên, suy yếu hay chưa có hướng đi rõ ràng.")
    st.write("• Biến động cho biết thị trường đang ổn định hay đang căng thẳng hơn mức thường thấy.")
    st.write("• Sức khỏe VN30 cho biết nhóm cổ phiếu vốn hóa lớn có đang cùng xác nhận diễn biến chung hay không.")
    st.write("• Mức độ phân hóa cho biết các cổ phiếu đang đi cùng nhau hay có sự khác biệt lớn về diễn biến.")
    st.write("• Tập trung biến động cho biết biến động đang lan rộng hay chủ yếu đến từ một nhóm nhỏ cổ phiếu.")
    st.info("Kết quả là công cụ hỗ trợ quan sát và quản trị danh mục. Không phải khuyến nghị mua bán và không thay thế việc xác định mục tiêu đầu tư hoặc mức chịu rủi ro của từng người dùng.")
