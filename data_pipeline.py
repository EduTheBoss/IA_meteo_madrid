import logging
import os
import re
from pathlib import Path
import numpy as np
import requests
import zipfile
import unicodedata
from io import BytesIO
import pandas as pd
from geopy.distance import geodesic
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sklearn.preprocessing import LabelEncoder

# Configure logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# === Configuration ===
BASE_DIR = Path(r"C:\Users\evpue\Desktop\Progra\AI\content")
CONFIG = {
    'aire': {
        'input_dir': BASE_DIR / 'input' / 'aire',
        'output_dir': BASE_DIR / 'output' / 'transformado_aire',
        'urls': {
            2019: "https://datos.madrid.es/egob/catalogo/201200-42-calidad-aire-horario.zip",
            2020: "https://datos.madrid.es/egob/catalogo/201200-10306316-calidad-aire-horario.zip",
            2021: "https://datos.madrid.es/egob/catalogo/201200-10306317-calidad-aire-horario.zip",
            2022: "https://datos.madrid.es/egob/catalogo/201200-10306318-calidad-aire-horario.zip",
            2023: "https://datos.madrid.es/egob/catalogo/201200-10306319-calidad-aire-horario.zip",
            2024: "https://datos.madrid.es/egob/catalogo/201200-10306320-calidad-aire-horario.zip",
        }
    },
    'meteo': {
        'input_dir': BASE_DIR / 'input' / 'meteo',
        'output_dir': BASE_DIR / 'output' / 'transformado_meteo',
        'urls': {
            #2001: "https://datos.madrid.es/egob/catalogo/201200-29-calidad-aire-horario.zip",
            #2002: "https://datos.madrid.es/egob/catalogo/201200-30-calidad-aire-horario.zip",
            #2003: "https://datos.madrid.es/egob/catalogo/201200-13-calidad-aire-horario.zip",
            #2004: "https://datos.madrid.es/egob/catalogo/201200-14-calidad-aire-horario.zip",
            #2005: "https://datos.madrid.es/egob/catalogo/201200-15-calidad-aire-horario.zip",
            #2006: "https://datos.madrid.es/egob/catalogo/201200-16-calidad-aire-horario.zip",
            #2007: "https://datos.madrid.es/egob/catalogo/201200-17-calidad-aire-horario.zip",
            #2008: "https://datos.madrid.es/egob/catalogo/201200-18-calidad-aire-horario.zip",
            #2009: "https://datos.madrid.es/egob/catalogo/201200-19-calidad-aire-horario.zip",
            #2010: "https://datos.madrid.es/egob/catalogo/201200-20-calidad-aire-horario.zip",
            #2011: "https://datos.madrid.es/egob/catalogo/201200-21-calidad-aire-horario.zip",
            #2012: "https://datos.madrid.es/egob/catalogo/201200-22-calidad-aire-horario.zip",
            #2013: "https://datos.madrid.es/egob/catalogo/201200-23-calidad-aire-horario.zip",
            #2014: "https://datos.madrid.es/egob/catalogo/201200-26-calidad-aire-horario.zip",
            #2015: "https://datos.madrid.es/egob/catalogo/201200-27-calidad-aire-horario.zip",
            #2016: "https://datos.madrid.es/egob/catalogo/201200-28-calidad-aire-horario.zip",
            #2017: "https://datos.madrid.es/egob/catalogo/201200-10306313-calidad-aire-horario.zip",
            #2018: "https://datos.madrid.es/egob/catalogo/201200-10306314-calidad-aire-horario.zip",
            2019: "https://datos.madrid.es/egob/catalogo/201200-42-calidad-aire-horario.zip",
            2020: "https://datos.madrid.es/egob/catalogo/201200-10306316-calidad-aire-horario.zip",
            2021: "https://datos.madrid.es/egob/catalogo/201200-10306317-calidad-aire-horario.zip",
            2022: "https://datos.madrid.es/egob/catalogo/201200-10306318-calidad-aire-horario.zip",
            2023: "https://datos.madrid.es/egob/catalogo/201200-10306319-calidad-aire-horario.zip",
            2024: "https://datos.madrid.es/egob/catalogo/201200-10306320-calidad-aire-horario.zip",
        }
    },
    'stations': {
        'aire': {
            'url': "https://datos.madrid.es/egob/catalogo/212629-1-estaciones-control-aire.csv",
            'dest': BASE_DIR / 'stations' / 'aire.csv'
        },
        'meteo': {
            'url': "https://datos.madrid.es/egob/catalogo/300360-1-meteorologicos-estaciones.csv",
            'dest': BASE_DIR / 'stations' / 'meteo.csv'
        }
    }
}

# === Setup HTTP session with retries ===
session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[502,503,504])
session.mount('https://', HTTPAdapter(max_retries=retries))

# === Utility functions ===

def ensure_dir(path: Path):
    """Ensure directory exists."""
    if not path.exists():
        print(f"Creating directory {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_and_extract(url: str, output_dir: Path, timeout: int = 30):
    """Download ZIP via session and extract CSVs."""
    print(f"Starting download from {url}")
    try:
        resp = session.get(url, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return
    with zipfile.ZipFile(BytesIO(resp.content)) as archive:
        for member in archive.infolist():
            if member.filename.lower().endswith('.csv'):
                target = output_dir / Path(member.filename).name
                print(f"Extracting {member.filename} to {target}")
                with archive.open(member) as src, target.open('wb') as dst:
                    dst.write(src.read())
    print(f"Finished extracting from {url} to {output_dir}")


def transform_wide_to_long(src_dir: Path, dest_dir: Path):
    """Convert H01..H24, V01..V24 to long format."""
    print(f"Transforming wide to long in {src_dir}")
    ensure_dir(dest_dir)
    for csv_file in tqdm(list(src_dir.glob('*.csv')), desc="Wide->Long"):
        print(f"Processing file {csv_file.name}")
        df = pd.read_csv(csv_file, sep=';', encoding='latin1')
        id_vars = ['PROVINCIA','MUNICIPIO','ESTACION','MAGNITUD','PUNTO_MUESTREO','ANO','MES','DIA']
        df_long = df.melt(id_vars=id_vars, value_vars=[f'H{h:02d}' for h in range(1,25)], var_name='HORA', value_name='VALOR_HORA')
        df_val = df.melt(id_vars=id_vars, value_vars=[f'V{h:02d}' for h in range(1,25)], var_name='HORA_VALID', value_name='VALIDACION')
        df_long['HORA'] = df_long['HORA'].str.replace('H','').astype(int)
        df_val['HORA'] = df_val['HORA_VALID'].str.replace('V','').astype(int)
        df_val_clean = df_val.drop(columns=['HORA_VALID'])
        print(f"Merging measurement and validation for {csv_file.name}")
        df_merged = pd.merge(df_long, df_val_clean, on=id_vars + ['HORA'], how='left')
        out = dest_dir / f"{csv_file.stem}_transformado.csv"
        print(f"Writing transformed data to {out}")
        df_merged.to_csv(out, index=False)
    print(f"Completed transformation for {src_dir}")


def pivot_meteo(src_dir: Path, dest_dir: Path):
    """Pivot meteorology by MAGNITUD."""
    print(f"Pivoting meteorology files in {src_dir}")
    ensure_dir(dest_dir)
    for csv_file in tqdm(list(src_dir.glob('*.csv')), desc="Pivot Meteo"):
        print(f"Pivoting {csv_file.name}")
        df = pd.read_csv(csv_file)
        idx_cols = ['PROVINCIA','MUNICIPIO','ESTACION','ANO','MES','DIA','HORA']
        df_piv = df.pivot_table(index=idx_cols, columns='MAGNITUD', values='VALOR_HORA').rename_axis(None, axis=1).reset_index()
        rename_map = {m: f'VALOR_HORA_{int(m)}' for m in df_piv.columns if isinstance(m,(int,float))}
        df_piv.rename(columns=rename_map, inplace=True)
        out = dest_dir / f"{csv_file.stem}_piv.csv"
        print(f"Writing pivoted data to {out}")
        df_piv.to_csv(out, index=False)
    print(f"Completed pivoting for {src_dir}")


def remove_magnitude(src_dir: Path, dest_dir: Path, mag: int = 80):
    ensure_dir(dest_dir)
    col = f'VALOR_HORA_{mag}'
    print(f"Removing magnitude {mag} columns from {src_dir}")
    for fpath in tqdm(list(src_dir.glob('*.csv')), desc=f"Drop {mag}"):
        df = pd.read_csv(fpath)
        if col in df.columns:
            print(f"Dropping column {col} in {fpath.name}")
            df.drop(columns=[col], inplace=True)
        df.to_csv(dest_dir / fpath.name, index=False)
    print(f"Completed removing magnitude {mag}")


def drop_columns(src_dir: Path, dest_dir: Path, cols: list):
    ensure_dir(dest_dir)
    print(f"Dropping columns {cols} from files in {src_dir}")
    for fpath in tqdm(list(src_dir.glob('*.csv')), desc="Drop cols"):
        df = pd.read_csv(fpath)
        df.drop(columns=[c for c in cols if c in df.columns], inplace=True)
        df.to_csv(dest_dir / fpath.name, index=False)
    print(f"Completed dropping columns for {src_dir}")


def filter_and_clean(src_dir: Path, dest_dir: Path, valid_flag: str = 'V'):
    ensure_dir(dest_dir)
    print(f"Filtering valid rows (flag={valid_flag}) in {src_dir}")
    for fpath in tqdm(list(src_dir.glob('*.csv')), desc="Filter valid"):
        df = pd.read_csv(fpath)
        if 'VALIDACION' in df.columns:
            df = df[df['VALIDACION'] == valid_flag]
            df.drop(columns=['VALIDACION'], inplace=True)
        df.to_csv(dest_dir / fpath.name, index=False)
    print(f"Completed filtering valid rows for {src_dir}")


def download_stations():
    print("Downloading station metadata")
    for key, info in CONFIG['stations'].items():
        dest = Path(info['dest'])
        ensure_dir(dest.parent)
        try:
            resp = session.get(info['url'], timeout=30)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            print(f"Downloaded stations {key} -> {dest}")
        except requests.RequestException as e:
            print(f"Error downloading stations {key}: {e}")
    print("Completed downloading station metadata")

def normalize_col(name: str) -> str:
    """Normalize column name to ASCII lowercase for matching."""
    return unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode().lower()


def dms_to_decimal(dms_str: str) -> float:
    """
    Convert a DMS-like string into decimal degrees.
    E.g. "40º 24' 52''N", '40°25′25.98"N', etc.
    Looks for the first three numbers and the direction letter.
    """
    if not isinstance(dms_str, str):
        raise ValueError(f"Expected string, got {type(dms_str)}")

    # Find all numbers (integers or decimals)
    parts = re.findall(r'\d+(?:\.\d+)?', dms_str)
    if len(parts) < 3:
        raise ValueError(f"Not enough numeric parts in DMS: {dms_str!r}")
    deg, mins, secs = map(float, parts[:3])

    # Find the direction letter
    m = re.search(r'([NSEOW])', dms_str, re.IGNORECASE)
    if not m:
        raise ValueError(f"No direction (N/S/E/O) found in DMS: {dms_str!r}")
    direction = m.group(1).upper()

    dec = deg + mins/60 + secs/3600
    if direction in ('S', 'O', 'W'):
        dec = -dec
    return dec

def build_station_mapping(aire_csv: Path, meteo_csv: Path, out_csv: Path):
    """
    Build a CSV mapping each air-quality station to its nearest meteorological station.
    """
    # 1) Read both station files with UTF-8 (so we don't mangle any symbols)
    df_a = pd.read_csv(aire_csv, sep=';', encoding='utf-8')
    df_m = pd.read_csv(meteo_csv, sep=';', encoding='utf-8')

    # 2) Auto-detect the station-code & lat/lon columns by name
    def find_col(df, keyword):
        for c in df.columns:
            if keyword in c.lower():
                return c
        return None

    code_a = find_col(df_a, 'corto')
    code_m = find_col(df_m, 'corto')
    lat_a  = find_col(df_a, 'latitud')
    lon_a  = find_col(df_a, 'longitud')
    lat_m  = find_col(df_m, 'latitud')
    lon_m  = find_col(df_m, 'longitud')

    if not all([code_a, code_m, lat_a, lon_a, lat_m, lon_m]):
        raise RuntimeError("Could not auto-detect all required columns.")

    # 3) Subset and rename for clarity
    df_a = df_a[[code_a, lat_a, lon_a]].rename(columns={
        code_a: 'ESTACION_CA', lat_a: 'LAT_A', lon_a: 'LON_A'
    })
    df_m = df_m[[code_m, lat_m, lon_m]].rename(columns={
        code_m: 'ESTACION_METEO', lat_m: 'LAT_M', lon_m: 'LON_M'
    })

    # 4) Zero-pad codes
    df_a['ESTACION_CA']     = df_a['ESTACION_CA'].astype(str).str.zfill(3)
    df_m['ESTACION_METEO']  = df_m['ESTACION_METEO'].astype(str).str.zfill(3)

    # 5) Parse any DMS strings to decimal floats
    for df, lat_col, lon_col in [
        (df_a, 'LAT_A', 'LON_A'),
        (df_m, 'LAT_M', 'LON_M'),
    ]:
        if df[lat_col].dtype == object:
            df[lat_col] = df[lat_col].apply(dms_to_decimal)
            df[lon_col] = df[lon_col].apply(dms_to_decimal)

    # 6) For each air station, find the nearest meteo station
    records = []
    for _, a in df_a.iterrows():
        distances = df_m.apply(
            lambda m: geodesic((a.LAT_A, a.LON_A), (m.LAT_M, m.LON_M)).km,
            axis=1
        )
        nearest = df_m.loc[distances.idxmin(), 'ESTACION_METEO']
        records.append({
            'ESTACION_CA':      a.ESTACION_CA,
            'ESTACION_METEO':   nearest
        })

    # 7) Write out the mapping
    mapping_df = pd.DataFrame(records)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_df.to_csv(out_path, index=False)
    print(f"Station mapping complete → {out_path}")


#Next function
def combine_datasets(ca_dir: Path, meteo_dir: Path, map_csv: Path, out_file: Path):
    print("Combining air quality and meteorology datasets")
    df_map = pd.read_csv(map_csv, dtype=str)
    all_meteo = pd.concat([pd.read_csv(f) for f in meteo_dir.glob('*.csv')], ignore_index=True)
    all_meteo['ESTACION'] = all_meteo['ESTACION'].astype(str).str.zfill(3)
    combined = []
    for f in tqdm(ca_dir.glob('*.csv'), desc='Combine'):
        print(f"Processing air file {f.name}")
        df_ca = pd.read_csv(f)
        df_ca['ESTACION'] = df_ca['ESTACION'].astype(str).str.zfill(3)
        for code, grp in df_ca.groupby('ESTACION'):
            metro_code = df_map.loc[df_map['ESTACION_CA']==code, 'ESTACION_METEO']
            if metro_code.empty:
                print(f"No mapping for CA station {code}, skipping")
                continue
            # rename the meteorology “ESTACION” to ESTACION_METEO right after subsetting
            df_m = all_meteo[all_meteo['ESTACION']==metro_code.iloc[0]].rename(
                columns={'ESTACION':'ESTACION_METEO'}
            )
            
            grp = grp.rename(columns={'ESTACION':'ESTACION_CA'})
            merged = pd.merge(grp, df_m, on=['ANO','MES','DIA','HORA'])
            if merged.empty:
                print(f"No matching meteorology for station {code} at same datetime, skipping")
                continue
            
            met_cols = [c for c in merged if c.startswith('VALOR_HORA_')]
            final = ['ESTACION_CA','MAGNITUD','ANO','MES','DIA','HORA','VALOR_HORA'] + met_cols
            combined.append(merged[final])
    df_final = pd.concat(combined, ignore_index=True)
    print(f"Writing combined dataset to {out_file}, total rows: {len(df_final)}")
    df_final.to_csv(out_file, index=False)
    print("Completed combining datasets")

def prepare_for_training(input_csv: Path, output_csv: Path):
    print("=== STEP 7: Preparar datos para entrenamiento ===")
    # 1) Carga del dataset combinado
    df = pd.read_csv(input_csv)
    print(f"Datos cargados: {len(df)} filas, {len(df.columns)} columnas")

    # 2) Mapear códigos de contaminante a nombres y eliminar MAGNITUD
    contaminant_map = {
        1:  "SO2",   6:  "CO",    7:  "NO",   8:  "NO2",
       9:  "PM2.5",10: "PM10",  12: "NOx",  14: "O3",
       20: "TOLUENO",30: "BENZENO",35: "ETILBENCENO",
       37: "MXYLENOS",38: "PXYLENOS",39: "OXYLENOS",
       42: "TCH",  43: "CH4",   44: "NMHC", 431:"MPX"
    }
    print("Mapeando columna MAGNITUD → CONTAMINANTE y eliminando MAGNITUD numérico...")
    df['CONTAMINANTE'] = df['MAGNITUD'].map(contaminant_map)
    df.drop(columns=['MAGNITUD'], inplace=True)
    print("→ Transformando CONTAMINANTE a IDs numéricos (LabelEncoder)...")
    le = LabelEncoder()
    df['CONT_ID'] = le.fit_transform(df['CONTAMINANTE'])

    # 3) Renombrar columnas meteorológicas a nombres legibles
    meteo_map = {
        'VALOR_HORA_1':  'Temperatura',
        'VALOR_HORA_6':  'Temp_Max',
        'VALOR_HORA_7':  'Temp_Min',
        'VALOR_HORA_8':  'Precipitacion',
        'VALOR_HORA_10': 'Radiacion',
        'VALOR_HORA_12': 'Humedad',
        'VALOR_HORA_14': 'Viento_Vel',
        'VALOR_HORA_20': 'Viento_Dir',
        'VALOR_HORA_30': 'Presion',
        'VALOR_HORA_35': 'Viento_Dir_Fina',
        'VALOR_HORA_42': 'Visibilidad',
        'VALOR_HORA_43': 'Evaporacion',
        'VALOR_HORA_44': 'Insolacion',
        'VALOR_HORA_9':  'Rafagas'
    }
    print("Renombrando columnas meteorológicas...")
    df.rename(columns=meteo_map, inplace=True)

    # 4) Ordenar por estación y fecha-hora para que la interpolación sea correcta
    print("Ordenando DataFrame por ESTACION_CA y fecha-hora...")
    df.sort_values(['ESTACION_CA','ANO','MES','DIA','HORA'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 5) Calcular porcentaje de faltantes y descartar variables muy incompletas
    missing_ratio = df.isna().mean()
    print("Ratio faltantes por columna (antes de dropear):")
    print(missing_ratio.sort_values(ascending=False).head(10))

    to_drop = missing_ratio[missing_ratio > 0.8].index.tolist()
    if to_drop:
        print(f"Eliminando columnas con >80% faltantes: {to_drop}")
        df.drop(columns=to_drop, inplace=True)

    # --- IMPORTANTE: recalculamos los ratios tras eliminar columnas ---
    missing_ratio = df.isna().mean()

    # 6a) Imputación simple (<15% faltante): interpolación + ffill/bfill
    simple_cols = missing_ratio[
        (missing_ratio > 0) & (missing_ratio <= 0.15)
    ].index.tolist()
    print(f"Columnas con <=15% faltante (a interpolar): {simple_cols}")
    for col in simple_cols:
        df[col] = df.groupby('ESTACION_CA')[col] \
                    .transform(lambda s: s.interpolate().ffill().bfill())

    # 6b) Imputación estacional (15–60% faltante): media por hora×mes + ffill/bfill
    seasonal_cols = missing_ratio[
        (missing_ratio > 0.15) & (missing_ratio <= 0.6)
    ].index.tolist()
    print(f"Columnas con 15–60% faltante (media estacional): {seasonal_cols}")
    for col in seasonal_cols:
        df[col] = (df
            .groupby(['ESTACION_CA','MES','HORA'])[col]
            .transform(lambda s: s.fillna(s.mean()).ffill().bfill())
        )

    # 7) Indicadores de qué valores fueron imputados
    print("Creando flags de valores imputados (_was_missing)...")
    for col in df.columns:
        if df[col].isna().any():
            df[f"{col}_was_missing"] = df[col].isna().astype(int)

    # 8) Relleno final de cualquier remanente
    df.fillna(0, inplace=True)
    
    return df
  
    
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based and cyclical features, drop unused cols, and reorder target."""

    print("=== STEP 8: Feature Engineering ===")

    # 8.1: Create a datetime column (optional but convenient)
    df['FECHA'] = pd.to_datetime({
        'year':  df['ANO'],
        'month': df['MES'],
        'day':   df['DIA'],
        'hour':  df['HORA']
    })

    # 8.2: Cyclical encoding of hour and month
    df['HORA_SIN'] = np.sin(2 * np.pi * df['HORA'] / 24)
    df['HORA_COS'] = np.cos(2 * np.pi * df['HORA'] / 24)
    df['MES_SIN']  = np.sin(2 * np.pi * df['MES']  / 12)
    df['MES_COS']  = np.cos(2 * np.pi * df['MES']  / 12)

    # day-of-week encoding (0=Mon,..6=Sun)
    df['DOW'] = df['FECHA'].dt.weekday
    df['DOW_SIN'] = np.sin(2 * np.pi * df['DOW'] / 7)
    df['DOW_COS'] = np.cos(2 * np.pi * df['DOW'] / 7)

    # optional: day-of-month encoding (1–31)
    df['DOM'] = df['FECHA'].dt.day
    df['DOM_SIN'] = np.sin(2 * np.pi * (df['DOM']-1) / 31)
    df['DOM_COS'] = np.cos(2 * np.pi * (df['DOM']-1) / 31)

    # 8.3: Drop raw date cols and pollutant name
    drop_feats = ['ANO','MES','DIA','HORA','FECHA','CONTAMINANTE','DOW','DOM']
    df.drop(columns=[c for c in drop_feats if c in df.columns], inplace=True)
    
    # 8.4: (Optional) One-hot encode pollutant if you prefer not to use CONT_ID alone
    df = pd.get_dummies(df, columns=['CONT_ID'], prefix='CONT')

    # 8.4: Final column order: target last
    cols = [c for c in df.columns if c != 'VALOR_HORA'] + ['VALOR_HORA']
    df = df[cols]

    print("Feature engineering complete. Final shape:", df.shape)
    return df


if __name__ == '__main__':
    # Step 1: Download data if not already present
    print("=== STEP 1: Downloading air quality data ===")
    aire_in = ensure_dir(CONFIG['aire']['input_dir'])
    existing_aire = list(aire_in.glob('*.csv'))
    if existing_aire and len(existing_aire) >= len(CONFIG['aire']['urls']):
        print(f"Air quality data already downloaded ({len(existing_aire)} files), skipping download.")
    else:
        for year, url in CONFIG['aire']['urls'].items():
            download_and_extract(url, aire_in)

    print("=== STEP 1: Downloading meteorology data ===")
    meteo_in = ensure_dir(CONFIG['meteo']['input_dir'])
    existing_meteo = list(meteo_in.glob('*.csv'))
    if existing_meteo and len(existing_meteo) >= len(CONFIG['meteo']['urls']):
        print(f"Meteorology data already downloaded ({len(existing_meteo)} files), skipping download.")
    else:
        for year, url in CONFIG['meteo']['urls'].items():
            download_and_extract(url, meteo_in)

    # Step 2: Transform wide to long only if not done
    print("=== STEP 2: Transforming wide to long format ===")
    aire_long_dir = ensure_dir(CONFIG['aire']['output_dir'])
    sample_aire_in = next(aire_in.glob('*.csv'), None)
    if sample_aire_in and any(aire_long_dir.glob(f"{sample_aire_in.stem}_transformado.csv")):
        print("Air quality already transformed to long format, skipping.")
    else:
        transform_wide_to_long(aire_in, aire_long_dir)

    meteo_long_dir = ensure_dir(CONFIG['meteo']['output_dir'])
    sample_meteo_in = next(meteo_in.glob('*.csv'), None)
    if sample_meteo_in and any(meteo_long_dir.glob(f"{sample_meteo_in.stem}_transformado.csv")):
        print("Meteorology already transformed to long format, skipping.")
    else:
        transform_wide_to_long(meteo_in, meteo_long_dir)

    # Step 3: Pivot & clean meteorology only if not done
    print("=== STEP 3: Pivoting and cleaning meteorology ===")
    meteo_piv_dir = BASE_DIR / 'output' / 'transformado_meteo_final'
    ensure_dir(meteo_piv_dir)
    sample_meteo_long = next(meteo_long_dir.glob('*_transformado.csv'), None)
    if sample_meteo_long and any(meteo_piv_dir.glob(f"{sample_meteo_long.stem}_piv.csv")):
        print("Meteorology already pivoted, skipping.")
    else:
        pivot_meteo(meteo_long_dir, meteo_piv_dir)
    meteo_no80_dir = BASE_DIR / 'output' / 'transformado_meteo_clean'
    if any(meteo_no80_dir.glob('*.csv')):
        print("Magnitude 80 already removed, skipping.")
    else:
        remove_magnitude(meteo_piv_dir, meteo_no80_dir, mag=80)

    # Step 4: Clean air quality
    print("=== STEP 4: Cleaning air quality data ===")
    aire_clean_dir = BASE_DIR / 'output' / 'transformado_aire_limpio'
    if any(aire_clean_dir.glob('*.csv')):
        print("Air quality clean already exists, skipping.")
    else:
        drop_columns(aire_long_dir, aire_clean_dir, ['PROVINCIA','MUNICIPIO','PUNTO_MUESTREO'])
    aire_final_dir = BASE_DIR / 'output' / 'transformado_aire_final'
    if any(aire_final_dir.glob('*.csv')):
        print("Air quality filtered already exists, skipping.")
    else:
        filter_and_clean(aire_clean_dir, aire_final_dir)

    # Step 5: Stations metadata and mapping
    print("=== STEP 5: Stations metadata and mapping ===")
    stations_dir = BASE_DIR / 'stations'
    mapping_file = stations_dir / 'mapping.csv'
    if mapping_file.exists():
        print("Station mapping already exists, skipping metadata download and mapping.")
    else:
        download_stations()
        build_station_mapping(stations_dir / 'aire.csv', stations_dir / 'meteo.csv', mapping_file)

    # Step 6: Combine datasets
    print("=== STEP 6: Combining datasets ===")
    final_dataset = BASE_DIR / 'output' / 'dataset_combinado_final.csv'
    if final_dataset.exists():
        print("Final combined dataset already exists, skipping combination.")
    else:
        combine_datasets(
            BASE_DIR / 'output' / 'transformado_aire_final',
            BASE_DIR / 'output' / 'transformado_meteo_clean',
            mapping_file,
            final_dataset
        )
    
    # STEP 7: Preparar datos para entrenamiento
    print("=== STEP 7: Preparar datos para entrenamiento ===")
    training_input  = final_dataset
    training_output = BASE_DIR / 'output' / 'dataset_listo_para_entrenar.csv'
    if training_output.exists():
        print(f"Training output already exists: {training_output}")
    else:
        df = prepare_for_training(training_input, training_output)
        de = feature_engineering(df)
        df.to_csv(training_output, index=False)
        print(f"→ Datos listos para entrenamiento en: {training_output}\n")