from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / 'data' / 'raw'


def raw_path(symbol):
    return RAW_DIR / f'{symbol.lower()}.parquet'


def load_raw(symbol):
    path = raw_path(symbol)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    return df.dropna(subset=['time']).sort_values('time').drop_duplicates('time', keep='last').reset_index(drop=True)


def available_symbols():
    if not RAW_DIR.exists():
        return []
    return sorted(p.stem.upper() for p in RAW_DIR.glob('*.parquet'))

def missing_symbols(symbols):
    available=set(available_symbols())
    return [s for s in symbols if s.upper() not in available]
