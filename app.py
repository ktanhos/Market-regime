from pathlib import Path
import sys
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.local_data import load_raw
from src.cached_breadth import build_cached_breadth

st.set_page_config(page_title='Trạng thái thị trường Việt Nam', page_icon='📊', layout='wide')

st.markdown('''<style>
:root{--good:#15803d;--bad:#dc2626;--warn:#ca8a04;--ink:#172033;--muted:#667085;--line:#e4e7ec}
.block-container{max-width:1280px;padding-top:1.5rem;padding-bottom:3rem}
.hero,.card{padding:1.3rem;border:1px solid var(--line);border-radius:16px;background:#fff;margin-bottom:1rem}
.hero h1{margin:0 0 .4rem;color:var(--ink)}.muted{color:var(--muted);line-height:1.6}.label{font-size:.8rem;color:var(--muted)}.value{font-size:1.4rem;font-weight:700;margin:.25rem 0}.good{color:var(--good)}.bad{color:var(--bad)}.warn{color:var(--warn)}
</style>''', unsafe_allow_html=True)

def cls(x):
    x=str(x).upper()
    if any(k in x for k in ['TÍCH CỰC','KHỎE','THẤP','HẠ NHIỆT']): return 'good'
    if any(k in x for k in ['SUY YẾU','YẾU','RỦI RO','ÁP LỰC','CAO']): return 'bad'
    return 'warn'

def features(df):
    x=df.copy(); c=x['close']
    x['strength']=(c.pct_change(63)*.4+c.pct_change(126)*.2+c.pct_change(189)*.2+c.pct_change(252)*.2)*100
    x['roro']=x['strength']-x['strength'].rolling(49,min_periods=49).mean()
    x['trend']=np.where(x['roro']>0,'TÍCH CỰC','SUY YẾU')
    lr=np.log(x['high']/x['low'])
    x['vol']=np.sqrt(lr.pow(2).rolling(22,min_periods=22).mean()/(4*np.log(2))*252)*100
    base=x['vol'].rolling(252,min_periods=60).mean(); sd=x['vol'].rolling(252,min_periods=60).std().replace(0,np.nan)
    z=(x['vol']-base)/sd
    x['stress']=np.select([z>=1.5,z>=.5,z<=-.5],['RẤT CAO','CAO','THẤP'],default='BÌNH THƯỜNG')
    return x

def regime(t,s):
    if t=='TÍCH CỰC' and s=='THẤP': return 'THỊ TRƯỜNG TÍCH CỰC','Xu hướng tốt và biến động thấp hơn thông thường.','Có thể duy trì mức rủi ro cao hơn trong giới hạn đã xác định.'
    if t=='SUY YẾU' and s in ['CAO','RẤT CAO']: return 'THỊ TRƯỜNG ĐANG CHỊU ÁP LỰC','Xu hướng suy yếu đi cùng biến động cao.','Ưu tiên giảm rủi ro, kiểm soát tỷ trọng và hạn chế đòn bẩy.'
    if t=='TÍCH CỰC' and s in ['CAO','RẤT CAO']: return 'TĂNG NHƯNG BIẾN ĐỘNG CAO','Xu hướng tích cực nhưng mức rủi ro đã tăng.','Duy trì tỷ trọng có chọn lọc và kiểm soát rủi ro.'
    return 'GIAI ĐOẠN CHUYỂN TIẾP','Xu hướng và biến động chưa cùng xác nhận một trạng thái rõ ràng.','Duy trì mức rủi ro vừa phải và chờ thêm tín hiệu xác nhận.'

@st.cache_data(show_spinner=False)
def load_all():
    idx=load_raw('VNINDEX')
    path=ROOT/'data'/'processed'/'vn30_metrics_history.parquet'
    m=pd.read_parquet(path) if path.exists() else None
    if m is not None: m.index=pd.to_datetime(m.index)
    return idx,m

idx,metrics=load_all()
st.markdown("<div class='hero'><h1>Trạng thái thị trường Việt Nam</h1><div class='muted'>Ứng dụng mô tả thị trường hiện tại từ xu hướng VNINDEX, mức biến động và cấu trúc của nhóm VN30. Dashboard chỉ đọc dữ liệu đã lưu và không gọi API khi mở.</div></div>",unsafe_allow_html=True)
if idx is None or idx.empty:
    st.error('Chưa có dữ liệu VNINDEX đã lưu trong data/raw.')
    st.stop()

x=features(idx); r=x.iloc[-1]; t=r['trend']; s=r['stress']; title,text,advice=regime(t,s)
b=build_cached_breadth()
last=pd.Timestamp(r['time']).strftime('%d/%m/%Y')

st.caption(f'Dữ liệu VNINDEX đến ngày {last}.')
st.markdown(f"<div class='card'><div class='label'>ĐÁNH GIÁ CHUNG</div><div class='value {cls(title)}'>{title}</div><div class='muted'>{text}</div></div>",unsafe_allow_html=True)

c1,c2,c3=st.columns(3)
for col,label,value,note in [(c1,'XU HƯỚNG VNINDEX',t,f'RORO: {r.roro:.2f}'),(c2,'MỨC BIẾN ĐỘNG',s,f'Parkinson: {r.vol:.2f}%'),(c3,'SỨC KHỎE VN30',b['breadth_state'],f"Điểm tổng hợp: {b['breadth_score']:.1f}/100")]:
    with col: st.markdown(f"<div class='card'><div class='label'>{label}</div><div class='value {cls(value)}'>{value}</div><div class='muted'>{note}</div></div>",unsafe_allow_html=True)

st.subheader('Cấu trúc rủi ro VN30')
if metrics is not None and len(metrics):
    m=metrics.iloc[-1]
    a,c,d,e=st.columns(4)
    a.metric('Phân hóa',f"{m.get('dispersion',np.nan):.2f}%",f"Phân vị 252 phiên: {m.get('dispersion_pct_252',np.nan):.0f}")
    c.metric('Tập trung biến động',f"{m.get('top5_risk_share',np.nan):.1f}%",f"Phân vị 252 phiên: {m.get('top5_risk_share_pct_252',np.nan):.0f}")
    d.metric('Số mã hiệu dụng',f"{m.get('effective_risk_names',np.nan):.1f}")
    e.metric('Dữ liệu hợp lệ',f"{b['valid_symbols']}/{b['symbols']}")
else:
    st.warning('Chưa có vn30_metrics_history.parquet.')

st.subheader('Quản trị danh mục')
st.markdown(f"<div class='card'><div class='value {cls(title)}'>{title}</div><div class='muted'>{advice}</div></div>",unsafe_allow_html=True)

with st.expander('Xem chi tiết VN30'):
    st.dataframe(b['details'],use_container_width=True,hide_index=True)
    st.dataframe(b['risk_table'],use_container_width=True,hide_index=True)
    if b['failed']: st.caption('Thiếu dữ liệu: '+', '.join(b['failed']))

st.caption('Ứng dụng hỗ trợ quan sát trạng thái thị trường và quản trị rủi ro. Không phải khuyến nghị mua bán.')
