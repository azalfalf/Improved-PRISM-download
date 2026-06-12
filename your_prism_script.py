# Ensure you run this once in a cell above if it's missing:
# !pip install rasterio requests --quiet

import os
import zipfile
import math
import requests
from io import BytesIO
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import rasterio

# ==========================================
# 📍 CONFIGURATIONS & PARAMETERS
# ==========================================
LATITUDE = 35.879129
LONGITUDE = -106.615240
ELEVATION = 1732  # meters

MASTER_DRIVE_PATH = "/content/drive/MyDrive/TEST_DATA.csv"
OUTPUT_PATH = "/content/drive/MyDrive/PROCESSED_ET_HU_DATA.csv"
TEMP_EXTRACT_DIR = "./prism_temp"

os.makedirs(TEMP_EXTRACT_DIR, exist_ok=True)

# Mount Google Drive safely
if not os.path.exists("/content/drive"):
    from google.colab import drive
    drive.mount("/content/drive")

# Calculate target date (Yesterday)
yesterday = datetime.now() - timedelta(days=1)
date_str = yesterday.strftime("%Y%m%d")  # Format: YYYYMMDD

# ==========================================
# 🔄 STEP 1: DOWNLOAD & RASTERIO EXTRACTION
# ==========================================
def extract_pixel_from_prism(variable, date_string, lat, lon):
    """
    Downloads the daily PRISM raster zip file, extracts the GeoTIFF, 
    and samples the exact coordinate pixel value using rasterio.
    """
    # 🟢 FIXED: Added the required forward slash right after the base url
    url = f"https://services.nacse.org/prism/data/get/us/4km/{variable}/{date_string}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    print(f"📥 Downloading PRISM {variable.upper()} raster for {date_string}...")
    response = requests.get(url, headers=headers, timeout=30)
    
    if response.status_code != 200:
        raise ConnectionError(f"Failed to fetch {variable}. PRISM HTTP Error: {response.status_code}")
        
    # Unpack zip archive in system memory cache
    with zipfile.ZipFile(BytesIO(response.content)) as z:
        raster_file_name = next(name for name in z.namelist() if name.endswith(('.bil', '.tif')))
        z.extract(raster_file_name, path=TEMP_EXTRACT_DIR)
        full_raster_path = os.path.join(TEMP_EXTRACT_DIR, raster_file_name)

    # Intersection query using Rasterio spatial coordinate map transforms
    with rasterio.open(full_raster_path) as src:
        coord_pair = [(lon, lat)]
        sampled_generator = src.sample(coord_pair)
        pixel_value = next(sampled_generator)
        
    # Delete temporary storage arrays
    os.remove(full_raster_path)
    return float(pixel_value[0])

# Try running live update pipeline; fall back gracefully to Drive database on error
try:
    fresh_tmax = extract_pixel_from_prism("tmax", date_str, LATITUDE, LONGITUDE)
    fresh_tmin = extract_pixel_from_prism("tmin", date_str, LATITUDE, LONGITUDE)
    fresh_ppt  = extract_pixel_from_prism("ppt", date_str, LATITUDE, LONGITUDE)
    fresh_vmax = extract_pixel_from_prism("vpdmax", date_str, LATITUDE, LONGITUDE)
    fresh_vmin = extract_pixel_from_prism("vpdmin", date_str, LATITUDE, LONGITUDE)
    
    new_row = pd.DataFrame({
        'Date': [pd.to_datetime(yesterday.date())],
        'ppt (mm)': [fresh_ppt],
        'tmin (degrees C)': [fresh_tmin],
        'tmax (degrees C)': [fresh_tmax],
        'vpdmin (hPa)': [fresh_vmin],
        'vpdmax (hPa)': [fresh_vmax],
        'soltotal (MJ/m^2/day)': [15.0]  # Standard fallback clear-sky placeholder constant
    })
    
    # 🟢 FIXED: Added safety header scanning checks within the file merger
    if os.path.exists(MASTER_DRIVE_PATH):
        with open(MASTER_DRIVE_PATH, "r") as f:
            peek = f.readline()
        
        if "date" in peek.lower() or "ppt" in peek.lower():
            df_master = pd.read_csv(MASTER_DRIVE_PATH)
        else:
            df_master = pd.read_csv(MASTER_DRIVE_PATH, skiprows=10)
            
        df_master.columns = df_master.columns.str.strip()
        df_master['Date'] = pd.to_datetime(df_master['Date'], format='mixed', errors='coerce')
        df = pd.concat([df_master, new_row], ignore_index=True).drop_duplicates(subset=['Date'], keep='last')
    else:
        df = new_row
        
    df.sort_values(by="Date", inplace=True)
    df.to_csv(MASTER_DRIVE_PATH, index=False)
    print("✅ Successfully synchronized yesterday's metrics via Rasterio.")

except Exception as e:
    print(f"\n⚠️ Live update skipped. Reason: {e}")
    print("🔄 Activating safe recovery routine using historical database...")
    
    if os.path.exists(MASTER_DRIVE_PATH):
        # 🟢 FIXED: Applied identical auto-detect logic here to eliminate KeyError: 'Date'
        with open(MASTER_DRIVE_PATH, "r") as f:
            peek = f.readline()
        
        if "date" in peek.lower() or "ppt" in peek.lower():
            df = pd.read_csv(MASTER_DRIVE_PATH)
        else:
            df = pd.read_csv(MASTER_DRIVE_PATH, skiprows=10)
            
        df.columns = df.columns.str.strip()
    else:
        raise FileNotFoundError(f"Critical Error: No historical file found at: {MASTER_DRIVE_PATH}")

# ==========================================
# 🔄 STEP 2: RUN COMPUTATION LOGIC PIPELINES
# ==========================================

# Safely track down column string positions dynamically
actual_date_col = next(c for c in df.columns if any(x in c.lower() for x in ["date", "name"]))
t_max = next(c for c in df.columns if "tmax" in c.lower())
t_min = next(c for c in df.columns if "tmin" in c.lower())
sol_col = next(c for c in df.columns if "sol" in c.lower())
vmax_c = next(c for c in df.columns if "vpdmax" in c.lower())
vmin_c = next(c for c in df.columns if "vpdmin" in c.lower())

df['Date'] = pd.to_datetime(df[actual_date_col], format="mixed")
df['DOY'] = df['Date'].dt.dayofyear

lat_rad = math.radians(LATITUDE)
patm = 101.3 * (((293 - 0.0065 * ELEVATION) / 293) ** 5.26)
gamma = 0.000665 * patm

eto = []
for idx, row in df.iterrows():
    try:
        doy = row["DOY"]
        tmax_c = row[t_max]
        tmin_c = row[t_min]
        tm = (tmax_c + tmin_c) / 2
        v_mean = 2.0

        dl = (4098 * (0.6108 * math.exp((17.27 * tm) / (tm + 237.3)))) / ((tm + 237.3) ** 2)
        dec = 0.409 * math.sin((2 * math.pi / 365) * doy - 1.39)
        sha = math.acos(-math.tan(lat_rad) * math.tan(dec))
        dr = 1 + 0.033 * math.cos((2 * math.pi / 365) * doy)
        ra = ((24 * 60 / math.pi) * 0.0820 * dr * (sha * math.sin(lat_rad) * math.sin(dec) + math.cos(lat_rad) * math.cos(dec) * math.sin(sha)))

        rso = (0.75 + (2e-5 * ELEVATION)) * ra
        rs = row[sol_col]
        rns = (1 - 0.23) * rs

        vpd_max = row[vmax_c]
        vpd_min = row[vmin_c]

        eo_tmax = 0.6108 * math.exp((17.27 * tmax_c) / (tmax_c + 237.3))
        eo_tmin = 0.6108 * math.exp((17.27 * tmin_c) / (tmin_c + 237.3))
        es = (eo_tmax + eo_tmin) / 2
        vpd_mean = (vpd_max + vpd_min) / 2
        ea = max(0.01, es - vpd_mean)

        tk_max = tmax_c + 273.16
        tk_min = tmin_c + 273.16
        fcd = max(0.05, min(1.0, 1.35 * (rs / max(0.001, rso)) - 0.35))

        rnl = (4.903e-9 * ((tk_max**4 + tk_min**4) / 2) * (0.34 - 0.14 * math.sqrt(ea)) * fcd)
        rn = rns - rnl

        eto_val = (0.408 * dl * rn + gamma * (900 / (tm + 273)) * v_mean * (es - ea)) / (dl + gamma * (1 + 0.34 * v_mean))
        eto.append(max(0.0, eto_val))
    except Exception as row_err:
        eto.append(np.nan)

df["ETo_mm_day"] = eto

# Vectorized 85/50 Heat Unit Calculation
tmax_f = (df[t_max] * 9/5) + 32.0
tmin_f = (df[t_min] * 9/5) + 32.0
tmax_bounded = np.clip(tmax_f, 50.0, 85.0)
tmin_bounded = np.clip(tmin_f, 50.0, 85.0)
tavg_bounded = (tmax_bounded + tmin_bounded) / 2.0

df["Daily_HU"] = np.maximum(0.0, tavg_bounded - 50.0)
df["Cumulative_HU"] = df["Daily_HU"].cumsum()

# Save final processed tracking files back to Drive
df.to_csv(OUTPUT_PATH, index=False)
print(f"🚀 Execution successful! Output sync complete at: {OUTPUT_PATH}")
