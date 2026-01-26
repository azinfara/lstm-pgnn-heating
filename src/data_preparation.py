# =============================================================
# COMPLETE DATA PREP + STRATIFIED SPLIT FOR LSTM-PGNN
# =============================================================
import os
import random
from glob import glob
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import joblib
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
CHUNK_DIR = "processed_Heating_chunks"
MM_DIR = "mm_data_pinn_with_schedule"
os.makedirs(MM_DIR, exist_ok=True)

# Define columns
schedule_col = "schedule"

time_series_cols = [
    "Outdoor temp", "Humidity", "Radiation",
    "hour", "dayofweek", "month", "is_weekend", "schedule"
]

static_num_cols = [
    "Total Area", "Total Volume", "HRUefficiency", "Total int gain",
    "ExtWin U", "ExtWin g", "WWR", "Q50", "ShadingRate",
    "Number_of_AHU", "Max_airflow", "ExtWall U", "ExtWall Area"
]

static_cat_cols = ["Type", "Orientation"]

TARGET_COL = "Total Heating"

required_cols = (
    time_series_cols
    + static_num_cols
    + static_cat_cols
    + [TARGET_COL, "RunID", "BuildingID"]
)

# -------------------------------------------------------------
# 0) DISCOVER CLEAN CHUNKS
# -------------------------------------------------------------
print("="*70)
print("STEP 0: DISCOVERING CHUNKS")
print("="*70)

all_chunks = sorted(glob(f"{CHUNK_DIR}/*.parquet"))
chunk_files = [f for f in all_chunks if "withNaNs" not in f]
print(f"Found {len(all_chunks)} total chunks")

valid_chunks = []
for f in tqdm(chunk_files, desc="Validating chunks"):
    table = pq.read_table(f)
    
    if schedule_col not in table.column_names:
        print(f"⚠️  {os.path.basename(f)} missing '{schedule_col}' column, skipping.")
        continue
    
    missing_cols = [c for c in required_cols if c not in table.column_names]
    if missing_cols:
        print(f"⚠️  {os.path.basename(f)} missing columns: {missing_cols}, skipping.")
        continue
    
    table = pq.read_table(f, columns=required_cols)
    df = table.to_pandas()
    
    n_nan_rows = df[required_cols].isna().any(axis=1).sum()
    if n_nan_rows > 0:
        print(f"⚠️  Skipping {os.path.basename(f)} due to {n_nan_rows} rows with NaNs")
        continue
    
    valid_chunks.append(f)

print(f"\n✅ Using {len(valid_chunks)} valid chunks for processing")

if not valid_chunks:
    raise RuntimeError("No valid chunks remain after filtering!")

# ============================================================
# 1) STREAMING SCALER FIT
# ============================================================
print("\n" + "="*70)
print("STEP 1: FITTING SCALERS (STREAMING)")
print("="*70)

ts_min = None
ts_max = None

static_scaler = StandardScaler()
y_scaler = StandardScaler()

cat_values = {col: set() for col in static_cat_cols}

for f in tqdm(valid_chunks, desc="Fitting scalers"):
    table = pq.read_table(f, columns=required_cols)
    df = table.to_pandas()
    df["Max_airflow"] = df["Max_airflow"] / 1000.0
    
    if df.empty:
        continue

    ts_block = df[time_series_cols].values.astype(float)
    cur_min = ts_block.min(axis=0)
    cur_max = ts_block.max(axis=0)

    if ts_min is None:
        ts_min, ts_max = cur_min, cur_max
    else:
        ts_min = np.minimum(ts_min, cur_min)
        ts_max = np.maximum(ts_max, cur_max)

    static_block = df[static_num_cols].values.astype(float)
    static_scaler.partial_fit(static_block)

    y_kwh = (df[TARGET_COL].values.astype(float) / 1000.0).reshape(-1, 1)
    y_scaler.partial_fit(y_kwh)

    for col in static_cat_cols:
        cat_values[col].update(df[col].astype(str).unique())

print(f"  TS min: {ts_min}")
print(f"  TS max: {ts_max}")

ts_min_safe = np.array(ts_min, dtype=float)
ts_max_safe = np.array(ts_max, dtype=float)
range_ = ts_max_safe - ts_min_safe
range_[range_ == 0] = 1.0

ts_scaler = MinMaxScaler()
ts_limits = np.vstack([ts_min_safe, ts_max_safe])
ts_scaler.fit(ts_limits)
ts_scaler.data_range_ = np.where(ts_scaler.data_range_ == 0, 1.0, ts_scaler.data_range_)
ts_scaler.scale_ = 1.0 / ts_scaler.data_range_

print("\nFitting OHE encoders:")
ohe_encoders = {}
for col in static_cat_cols:
    cats = sorted(list(cat_values[col]))
    df_ohe = pd.DataFrame({col: cats})
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(df_ohe[[col]])
    ohe_encoders[col] = enc
    print(f"  {col}: {len(cats)} categories → {cats}")

joblib.dump(ts_scaler, "scaler_ts_pinn_with_schedule.pkl")
joblib.dump(static_scaler, "scaler_static_pinn_with_schedule.pkl")
joblib.dump(ohe_encoders, "ohe_encoder_pinn_with_schedule.pkl")
joblib.dump(y_scaler, "scaler_y_pinn_with_schedule.pkl")

print("\n✅ Saved all scalers")

# ============================================================
# 2) DETERMINE VALID FULL-YEAR RUN IDs (8760 ROWS)
# ============================================================
print("\n" + "="*70)
print("STEP 2: FINDING VALID 8760-HOUR RUNS")
print("="*70)

run_lengths = defaultdict(int)
building_of_run = {}
run_building_type = {}

type_stats = defaultdict(lambda: {'total': 0, 'with_schedule': 0})

for f in tqdm(valid_chunks, desc="Counting run lengths"):
    table = pq.read_table(f, columns=["RunID", "BuildingID", "Type", schedule_col])
    df = table.to_pandas()
    
    for runid, g in df.groupby("RunID"):
        run_lengths[runid] += len(g)
        building_of_run[runid] = g["BuildingID"].iloc[0]
        run_building_type[runid] = g["Type"].iloc[0]

total_runs_8760 = 0
for runid, length in run_lengths.items():
    if length == 8760:
        total_runs_8760 += 1
        btype = run_building_type[runid]
        type_stats[btype]['total'] += 1

valid_runids = sorted([r for r in run_lengths if run_lengths[r] == 8760])
valid_runids_set = set(valid_runids)

print(f"\n📊 Run Summary:")
print(f"  Total runs with 8760 hours: {total_runs_8760}")
print(f"  Valid runs: {len(valid_runids)}")

print(f"\n📊 By Building Type:")
for btype in sorted(type_stats.keys()):
    stats = type_stats[btype]
    print(f"  {btype:15s}: {stats['total']:3d} runs")

if not valid_runids:
    raise RuntimeError("No valid RunID with 8760 rows!")

total_rows = len(valid_runids) * 8760
print(f"\nTotal rows to write: {total_rows:,}")

# ============================================================
# 3) CREATE MEMMAPS
# ============================================================
print("\n" + "="*70)
print("STEP 3: CREATING MEMMAPS")
print("="*70)

ohe_dim = sum(enc.categories_[0].size for enc in ohe_encoders.values())
n_ts = len(time_series_cols)
n_static_num = len(static_num_cols)
n_static = n_static_num + ohe_dim
n_feat = n_static + n_ts

X_path = os.path.join(MM_DIR, "X_full.dat")
Y_path = os.path.join(MM_DIR, "y_full.dat")

mm_x = np.memmap(X_path, dtype="float32", mode="w+", shape=(total_rows, n_feat))
mm_y = np.memmap(Y_path, dtype="float32", mode="w+", shape=(total_rows,))

print(f"Feature dimensions:")
print(f"  Static numeric: {n_static_num}")
print(f"  Static OHE:     {ohe_dim}")
print(f"  Static total:   {n_static}")
print(f"  Time series:    {n_ts}")
print(f"  Total features: {n_feat}")
print(f"\nMemmap shapes: X={mm_x.shape}, y={mm_y.shape}")

# ============================================================
# 4) WRITE NORMALIZED DATA TO MEMMAPS
# ============================================================
print("\n" + "="*70)
print("STEP 4: WRITING DATA TO MEMMAPS")
print("="*70)

run_to_rows = {}
current = 0

for f in tqdm(valid_chunks, desc="Writing memmaps"):
    table = pq.read_table(f, columns=required_cols)
    df = table.to_pandas()
    df["Max_airflow"] = df["Max_airflow"] / 1000.0
    
    if df.empty:
        continue

    for runid, g in df.groupby("RunID"):
        if runid not in valid_runids_set:
            continue
        if len(g) != 8760:
            continue

        ts_scaled = ts_scaler.transform(g[time_series_cols])

        y_kwh = (g[TARGET_COL].values.astype(float) / 1000.0).reshape(-1, 1)
        y_scaled = y_scaler.transform(y_kwh).ravel().astype(np.float32)

        df_static0 = g.iloc[[0]][static_num_cols + static_cat_cols].copy()
        num_part = df_static0[static_num_cols].values.astype(np.float32)

        cat_parts = []
        for col in static_cat_cols:
            enc = ohe_encoders[col]
            cat_vals = enc.transform(df_static0[[col]].astype(str))
            cat_parts.append(cat_vals.astype(np.float32))
        cat_part = np.hstack(cat_parts) if cat_parts else np.zeros((1, 0), dtype=np.float32)

        static_combined = np.hstack([num_part, cat_part])
        static_scaled = static_scaler.transform(static_combined[:, :n_static_num])
        static_full = np.hstack([static_scaled, cat_part])

        static_block = np.repeat(static_full, len(g), axis=0)

        X_block = np.hstack([static_block, ts_scaled]).astype(np.float32)

        start = current
        end = current + len(g)
        mm_x[start:end] = X_block
        mm_y[start:end] = y_scaled

        run_to_rows[runid] = (start, end)
        current = end

mm_x.flush()
mm_y.flush()

print(f"\n✅ Wrote {current:,} rows to memmaps")

assert current == total_rows, f"Row mismatch: wrote {current}, expected {total_rows}"
print(f"✅ Row count validated: {current:,} == {total_rows:,}")

# ============================================================
# 5) BUILD BUILDING TYPE MAP
# ============================================================
print("\n" + "="*70)
print("STEP 5: BUILDING TYPE MAP")
print("="*70)

building_type = {}
for f in valid_chunks:
    table = pq.read_table(f, columns=["BuildingID", "Type"])
    df = table.to_pandas()
    for b, g in df.groupby("BuildingID"):
        if b not in building_type:
            building_type[b] = str(g["Type"].iloc[0])

print(f"✅ Collected {len(building_type)} buildings")

# ============================================================
# 6) STRATIFIED SPLIT BY TYPE + SIZE
# ============================================================
print("\n" + "="*70)
print("STEP 6: STRATIFIED TRAIN/VAL/TEST SPLIT")
print("="*70)

n_num = len(static_num_cols)
idx_area = static_num_cols.index("Total Area")

def get_building_info(runids):
    """Extract building type, size, and associated runs"""
    building_info = {}
    runids_set = set(runids)
    
    for runid in runids:
        bid = building_of_run[runid]
        
        if bid not in building_info:
            start, _ = run_to_rows[runid]
            static_scaled = mm_x[start, :n_static]
            static_num_scaled = static_scaled[:n_num]
            static_num_unscaled = static_num_scaled * np.sqrt(static_scaler.var_ + 1e-12) + static_scaler.mean_
            area = static_num_unscaled[idx_area]
            
            building_info[bid] = {
                'type': building_type.get(bid, 'Unknown'),
                'area': area,
                'runs': []
            }
        
        building_info[bid]['runs'].append(runid)
    
    return building_info

building_info = get_building_info(valid_runids)
print(f"Total unique buildings: {len(building_info)}")

all_areas = [info['area'] for info in building_info.values()]
p33 = np.percentile(all_areas, 33)
p67 = np.percentile(all_areas, 67)

print(f"\nSize bins:")
print(f"  Small:  < {p33:.0f} m²")
print(f"  Medium: {p33:.0f} - {p67:.0f} m²")
print(f"  Large:  > {p67:.0f} m²")

def get_size_category(area):
    if area < p33:
        return 'small'
    elif area < p67:
        return 'medium'
    else:
        return 'large'

groups = defaultdict(list)
for bid, info in building_info.items():
    btype = info['type']
    size_cat = get_size_category(info['area'])
    groups[(btype, size_cat)].append(bid)

print(f"\nFound {len(groups)} (Type, Size) groups:")
for key, buildings in sorted(groups.items()):
    print(f"  {key[0]:15s} {key[1]:8s}: {len(buildings):3d} buildings")

random.seed(42)
train_buildings, val_buildings, test_buildings = [], [], []

print("\nSplitting each group:")
for key, buildings in sorted(groups.items()):
    btype, size_cat = key
    n = len(buildings)
    
    if n < 3:
        train_buildings.extend(buildings)
        print(f"  {btype:15s} {size_cat:8s}: {n} → all train (too few)")
        continue
    
    random.shuffle(buildings)
    n_train = max(1, int(0.70 * n))
    n_val = max(1, int(0.15 * n))
    
    train_buildings.extend(buildings[:n_train])
    val_buildings.extend(buildings[n_train:n_train+n_val])
    test_buildings.extend(buildings[n_train+n_val:])
    
    print(f"  {btype:15s} {size_cat:8s}: {n:3d} → {n_train}/{n_val}/{n-n_train-n_val}")

train_runids = [r for bid in train_buildings for r in building_info[bid]['runs']]
val_runids = [r for bid in val_buildings for r in building_info[bid]['runs']]
test_runids = [r for bid in test_buildings for r in building_info[bid]['runs']]

print(f"\n📊 Final Split:")
print(f"  Train: {len(train_buildings)} buildings, {len(train_runids)} runs")
print(f"  Val:   {len(val_buildings)} buildings, {len(val_runids)} runs")
print(f"  Test:  {len(test_buildings)} buildings, {len(test_runids)} runs")

# ============================================================
# 7) DEFINE ALL COLUMN INDICES
# ============================================================
print("\n" + "="*70)
print("STEP 7: COLUMN INDICES")
print("="*70)

idx_Tout = time_series_cols.index("Outdoor temp")
idx_Humidity = time_series_cols.index("Humidity")
idx_Rad = time_series_cols.index("Radiation")
idx_hour = time_series_cols.index("hour")
idx_dayofweek = time_series_cols.index("dayofweek")
idx_month = time_series_cols.index("month")
idx_is_weekend = time_series_cols.index("is_weekend")
idx_Schedule = time_series_cols.index(schedule_col)

idx_area = static_num_cols.index("Total Area")
idx_vol = static_num_cols.index("Total Volume")
idx_HRU = static_num_cols.index("HRUefficiency")
idx_intgain = static_num_cols.index("Total int gain")
idx_Uwin = static_num_cols.index("ExtWin U")
idx_gwin = static_num_cols.index("ExtWin g")
idx_WWR = static_num_cols.index("WWR")
idx_Q50 = static_num_cols.index("Q50")
idx_shading = static_num_cols.index("ShadingRate")
idx_NumAHU = static_num_cols.index("Number_of_AHU")
idx_MaxAir = static_num_cols.index("Max_airflow")
idx_Uwall = static_num_cols.index("ExtWall U")
idx_Awall = static_num_cols.index("ExtWall Area")

print("Time series indices:")
print(f"  Tout={idx_Tout}, Humidity={idx_Humidity}, Rad={idx_Rad}")
print(f"  hour={idx_hour}, dayofweek={idx_dayofweek}, month={idx_month}")
print(f"  is_weekend={idx_is_weekend}, Schedule={idx_Schedule}")

print("\nStatic indices:")
print(f"  Area={idx_area}, Vol={idx_vol}, Q50={idx_Q50}")
print(f"  Uwall={idx_Uwall}, Awall={idx_Awall}, WWR={idx_WWR}")
print(f"  Uwin={idx_Uwin}, gwin={idx_gwin}, shading={idx_shading}")
print(f"  intgain={idx_intgain}, MaxAir={idx_MaxAir}, HRU={idx_HRU}")

# ============================================================
# 8) SAVE EVERYTHING
# ============================================================
print("\n" + "="*70)
print("STEP 8: SAVING ALL FILES")
print("="*70)

metadata = {
    "run_to_rows": run_to_rows,
    "building_of_run": building_of_run,
    "valid_runids": valid_runids,
    "n_features": n_feat,
    "n_ts_features": n_ts,
    "n_static_features": n_static,
    "n_static_num": n_static_num,
    "time_series_cols": time_series_cols,
    "static_num_cols": static_num_cols,
    "static_cat_cols": static_cat_cols,
    "target_col": TARGET_COL,
    "schedule_col": schedule_col,
}
joblib.dump(metadata, "metadata_pinn_with_schedule.pkl")
print("✅ Saved: metadata_pinn_with_schedule.pkl")

training_package = {
    "train_runids": train_runids,
    "val_runids": val_runids,
    "test_runids": test_runids,
    "run_to_rows": run_to_rows,
    "building_of_run": building_of_run,
    "building_type": building_type,
    "building_info": dict(building_info),
    "n_features": n_feat,
    "n_ts": n_ts,
    "n_static": n_static,
    "n_num": n_num,
    "time_series_cols": time_series_cols,
    "static_num_cols": static_num_cols,
    "static_cat_cols": static_cat_cols,
    "schedule_col": schedule_col,
    "target_col": TARGET_COL,
    "idx_Tout": idx_Tout,
    "idx_Humidity": idx_Humidity,
    "idx_Rad": idx_Rad,
    "idx_hour": idx_hour,
    "idx_dayofweek": idx_dayofweek,
    "idx_month": idx_month,
    "idx_is_weekend": idx_is_weekend,
    "idx_Schedule": idx_Schedule,
    "idx_area": idx_area,
    "idx_vol": idx_vol,
    "idx_HRU": idx_HRU,
    "idx_intgain": idx_intgain,
    "idx_Uwin": idx_Uwin,
    "idx_gwin": idx_gwin,
    "idx_WWR": idx_WWR,
    "idx_Q50": idx_Q50,
    "idx_shading": idx_shading,
    "idx_NumAHU": idx_NumAHU,
    "idx_MaxAir": idx_MaxAir,
    "idx_Uwall": idx_Uwall,
    "idx_Awall": idx_Awall,
    "mm_dir": MM_DIR,
    "total_rows": total_rows,
}
joblib.dump(training_package, "training_package_with_schedule.pkl")
print("✅ Saved: training_package_with_schedule.pkl")

scalers = {
    "ts_scaler": ts_scaler,
    "static_scaler": static_scaler,
    "y_scaler": y_scaler,
    "ohe_encoders": ohe_encoders,
}
joblib.dump(scalers, "scalers_with_schedule.pkl")
print("✅ Saved: scalers_with_schedule.pkl")

joblib.dump(train_runids, "train_runids_pinn_stratified_with_schedule.pkl")
joblib.dump(val_runids, "val_runids_pinn_stratified_with_schedule.pkl")
joblib.dump(test_runids, "test_runids_pinn_stratified_with_schedule.pkl")
print("✅ Saved: train/val/test split files")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*70)
print("✅ DATA PREP COMPLETE")
print("="*70)

print(f"\n📁 Output files:")
print(f"  Memmaps:  {MM_DIR}/X_full.dat, y_full.dat")
print(f"  Metadata: metadata_pinn_with_schedule.pkl")
print(f"  Package:  training_package_with_schedule.pkl")
print(f"  Scalers:  scalers_with_schedule.pkl")
print(f"  Splits:   train/val/test_runids_pinn_stratified_with_schedule.pkl")

print(f"\n📊 Data summary:")
print(f"  Valid runs:  {len(valid_runids)}")
print(f"  Total rows:  {total_rows:,}")
print(f"  Features:    {n_feat} ({n_static} static + {n_ts} TS)")
print(f"  Train/Val/Test: {len(train_runids)}/{len(val_runids)}/{len(test_runids)} runs")

print(f"\n🏗️ Building types:")
type_counts = Counter(building_type.values())
for t, c in sorted(type_counts.items()):
    print(f"  {t:15s}: {c:3d} buildings")