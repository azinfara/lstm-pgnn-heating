# ================================================================
# LSTM-PGNN V2.2 BASELINE TRAINING SCRIPT
# ================================================================
import os
import joblib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm.auto import tqdm

# ================================================================
# SETUP
# ================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ================================================================
# 1. LOAD DATA FROM DATA PREP OUTPUT
# ================================================================
print("\n" + "="*70)
print("STEP 1: LOADING DATA")
print("="*70)

pkg = joblib.load("training_package_with_schedule.pkl")

train_runids = pkg["train_runids"]
val_runids = pkg["val_runids"]
test_runids = pkg["test_runids"]
run_to_rows = pkg["run_to_rows"]
building_of_run = pkg["building_of_run"]
building_type = pkg["building_type"]

n_features = pkg["n_features"]
n_ts = pkg["n_ts"]
n_static = pkg["n_static"]
n_num = pkg["n_num"]

time_series_cols = pkg["time_series_cols"]
static_num_cols = pkg["static_num_cols"]
MM_DIR = pkg["mm_dir"]
total_rows = pkg["total_rows"]

# Column indices
idx_Tout = pkg["idx_Tout"]
idx_Rad = pkg["idx_Rad"]
idx_Schedule = pkg["idx_Schedule"]
idx_area = pkg["idx_area"]
idx_vol = pkg["idx_vol"]
idx_HRU = pkg["idx_HRU"]
idx_intgain = pkg["idx_intgain"]
idx_Uwin = pkg["idx_Uwin"]
idx_gwin = pkg["idx_gwin"]
idx_WWR = pkg["idx_WWR"]
idx_Q50 = pkg["idx_Q50"]
idx_shading = pkg["idx_shading"]
idx_NumAHU = pkg["idx_NumAHU"]
idx_MaxAir = pkg["idx_MaxAir"]
idx_Uwall = pkg["idx_Uwall"]
idx_Awall = pkg["idx_Awall"]

print(f"✅ Loaded training package")
print(f"   Train: {len(train_runids)} runs")
print(f"   Val:   {len(val_runids)} runs")
print(f"   Test:  {len(test_runids)} runs")

# ================================================================
# 2. LOAD SCALERS
# ================================================================
print("\n" + "="*70)
print("STEP 2: LOADING SCALERS")
print("="*70)

scalers = joblib.load("scalers_with_schedule.pkl")
ts_scaler = scalers["ts_scaler"]
static_scaler = scalers["static_scaler"]
y_scaler = scalers["y_scaler"]

print(f"✅ Loaded scalers")

# ================================================================
# 3. LOAD MEMMAPS
# ================================================================
print("\n" + "="*70)
print("STEP 3: LOADING MEMMAPS")
print("="*70)

X_mem = np.memmap(os.path.join(MM_DIR, "X_full.dat"), dtype="float32", mode="r", shape=(total_rows, n_features))
y_mem = np.memmap(os.path.join(MM_DIR, "y_full.dat"), dtype="float32", mode="r", shape=(total_rows,))

print(f"✅ Memmaps loaded: X={X_mem.shape}, y={y_mem.shape}")

# ================================================================
# 4. PREPARE SCALER TENSORS
# ================================================================
print("\n" + "="*70)
print("STEP 4: PREPARING SCALER TENSORS")
print("="*70)

ts_data_min = torch.tensor(ts_scaler.data_min_, dtype=torch.float32, device=device).view(1, 1, -1)
ts_data_range = torch.tensor(ts_scaler.data_range_, dtype=torch.float32, device=device).view(1, 1, -1)

static_mean_num = torch.tensor(static_scaler.mean_, dtype=torch.float32, device=device).view(1, -1)
static_std_num = torch.tensor(np.sqrt(static_scaler.var_ + 1e-12), dtype=torch.float32, device=device).view(1, -1)

y_mean = torch.tensor(y_scaler.mean_[0], dtype=torch.float32, device=device)
y_std = torch.tensor(np.sqrt(y_scaler.var_[0] + 1e-12), dtype=torch.float32, device=device)

print(f"✅ Scaler tensors ready")

# ================================================================
# 5. LOSS HYPERPARAMETERS
# ================================================================
LAMBDA_DELTA = 1e-3
LAMBDA_KREG = 1e-3

# ================================================================
# 6. DATASET & DATALOADER
# ================================================================
print("\n" + "="*70)
print("STEP 5: CREATING DATASETS")
print("="*70)

class RunDataset(Dataset):
    def __init__(self, runids, run_to_rows, X_mem, y_mem, n_static, n_ts):
        self.runids = list(runids)
        self.run_to_rows = run_to_rows
        self.X_mem = X_mem
        self.y_mem = y_mem
        self.n_static = n_static
        self.n_ts = n_ts
        self.T = 8760

    def __len__(self):
        return len(self.runids)

    def __getitem__(self, idx):
        runid = self.runids[idx]
        start, end = self.run_to_rows[runid]
        X_block = self.X_mem[start:end]
        y_block = self.y_mem[start:end]
        
        static_scaled = X_block[0, :self.n_static]
        ts_scaled = X_block[:, self.n_static:]
        
        return {
            "runid": runid,
            "X_ts": torch.tensor(ts_scaled, dtype=torch.float32),
            "static": torch.tensor(static_scaled, dtype=torch.float32),
            "y": torch.tensor(y_block.reshape(self.T, 1), dtype=torch.float32),
        }

def pinn_collate(batch):
    B = len(batch)
    T = batch[0]["X_ts"].shape[0]
    
    X_ts = torch.stack([b["X_ts"] for b in batch], dim=0)
    static = torch.stack([b["static"] for b in batch], dim=0)
    y = torch.stack([b["y"] for b in batch], dim=0)
    runids = [b["runid"] for b in batch]
    mask = torch.ones(B, T, 1, dtype=torch.float32)
    
    return {"runids": runids, "X_ts": X_ts, "static": static, "y": y, "mask": mask}

train_dataset = RunDataset(train_runids, run_to_rows, X_mem, y_mem, n_static, n_ts)
val_dataset = RunDataset(val_runids, run_to_rows, X_mem, y_mem, n_static, n_ts)
test_dataset = RunDataset(test_runids, run_to_rows, X_mem, y_mem, n_static, n_ts)

BATCH_SIZE = 4
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pinn_collate)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pinn_collate)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pinn_collate)

print(f"✅ Datasets ready: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

# ================================================================
# 7. MODEL DEFINITION
# ================================================================
print("\n" + "="*70)
print("STEP 6: DEFINING MODEL")
print("="*70)

class StaticEncoder(nn.Module):
    def __init__(self, static_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(static_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 4)
        )
    
    def forward(self, static):
        logits = self.net(static)
        k_env = torch.nn.functional.softplus(logits[:, 0:1]) + 0.3
        k_int = torch.nn.functional.softplus(logits[:, 1:2]) + 0.3
        k_sol = torch.nn.functional.softplus(logits[:, 2:3]) + 0.3
        k_vent = torch.nn.functional.softplus(logits[:, 3:4]) + 0.3
        return k_env, k_int, k_sol, k_vent


class LSTMPINNHeatingV2_2_Enhanced(nn.Module):
    def __init__(self, ts_dim, static_dim, hidden_lstm=64, num_layers=1, 
                 bidirectional=False, dropout=0.0, setpoint=22.0, 
                 correction_cap=5.0, exposure_factor=20.0, ua_multiplier=1.2,
                 schedule_weight=1.0):
        super().__init__()
        self.ts_dim = ts_dim
        self.static_dim = static_dim
        self.setpoint = setpoint
        self.correction_cap = correction_cap
        self.exposure_factor = exposure_factor
        self.ua_multiplier = ua_multiplier
        self.schedule_weight = schedule_weight
        self.hidden_lstm = hidden_lstm
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.static_enc = StaticEncoder(static_dim, hidden=64)
        
        self.lstm = nn.LSTM(
            input_size=ts_dim, 
            hidden_size=hidden_lstm,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_size = hidden_lstm * 2 if bidirectional else hidden_lstm
        
        self.head_corr = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_lstm),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_lstm, 1)
        )

    def forward(self, batch):
        X_ts = batch["X_ts"].to(device)
        static = batch["static"].to(device)
        B, T, _ = X_ts.shape

        # Unscale TS
        X_unscaled = X_ts * ts_data_range + ts_data_min
        T_out = X_unscaled[..., idx_Tout:idx_Tout+1]
        rad = X_unscaled[..., idx_Rad:idx_Rad+1]
        schedule = X_unscaled[..., idx_Schedule:idx_Schedule+1]
        occupancy_factor = schedule.clamp(0, 1)

        # Unscale static numeric
        static_num_scaled = static[:, :n_num]
        s = static_num_scaled * static_std_num + static_mean_num

        Total_Area = s[:, idx_area:idx_area+1]
        Volume = s[:, idx_vol:idx_vol+1]
        Awall = s[:, idx_Awall:idx_Awall+1]
        WWR = s[:, idx_WWR:idx_WWR+1].clamp(0, 1)
        Uwall = s[:, idx_Uwall:idx_Uwall+1]
        Uwin = s[:, idx_Uwin:idx_Uwin+1]
        gwin = s[:, idx_gwin:idx_gwin+1]
        shading = s[:, idx_shading:idx_shading+1].clamp(0, 1)
        int_g = s[:, idx_intgain:idx_intgain+1]
        MaxAir_Ls = s[:, idx_MaxAir:idx_MaxAir+1]
        Q50 = s[:, idx_Q50:idx_Q50+1]

        # Physics calculations
        A_opaque = Awall
        A_window = (WWR * Awall) / (1 - WWR + 1e-6)
        
        UA_walls = Uwall * A_opaque + Uwin * A_window
        UA_envelope = UA_walls * self.ua_multiplier
        
        Envelope_Area = Awall * 1.4
        n50 = (Q50 * Envelope_Area) / (Volume + 1e-6)
        ACH_natural = n50 / self.exposure_factor
        V_infilt = (Volume * ACH_natural) / 3600.0
        m_infilt = V_infilt * 1.2
        
        m_vent_max = (MaxAir_Ls / 1000.0) * 1.2
        m_vent_base = m_vent_max * 0.3
        m_vent_variable = m_vent_max * 0.7

        INT_max = int_g * Total_Area
        INT_base = INT_max * 0.1
        INT_variable = INT_max * 0.9

        SOLCOEFF = gwin * A_window * (1 - shading)

        UA_env_t = UA_envelope.unsqueeze(1).expand(B, T, 1)
        m_infilt_t = m_infilt.unsqueeze(1).expand(B, T, 1)
        
        m_vent_t = m_vent_base.unsqueeze(1).expand(B, T, 1) + \
                   m_vent_variable.unsqueeze(1).expand(B, T, 1) * occupancy_factor * self.schedule_weight
        
        INT_t = INT_base.unsqueeze(1).expand(B, T, 1) + \
                INT_variable.unsqueeze(1).expand(B, T, 1) * occupancy_factor * self.schedule_weight
        
        SOL_t = SOLCOEFF.unsqueeze(1).expand(B, T, 1)

        k_env, k_int, k_sol, k_vent = self.static_enc(static)
        k_env_t = k_env.unsqueeze(1).expand(B, T, 1)
        k_int_t = k_int.unsqueeze(1).expand(B, T, 1)
        k_sol_t = k_sol.unsqueeze(1).expand(B, T, 1)
        k_vent_t = k_vent.unsqueeze(1).expand(B, T, 1)

        T_set = torch.full_like(T_out, self.setpoint)
        dT = torch.relu(T_set - T_out)

        Q_envelope = (k_env_t * UA_env_t * dT) / 1000.0
        Q_infiltration = k_env_t * m_infilt_t * 1.0 * dT
        Q_vent = k_vent_t * m_vent_t * 1.0 * dT
        Q_int = (k_int_t * INT_t) / 1000.0
        Q_sol = (k_sol_t * SOL_t * rad) / 1000.0

        Q_phys = torch.relu(Q_envelope + Q_infiltration + Q_vent - Q_int - Q_sol)

        # LSTM correction
        h_lstm, _ = self.lstm(X_ts)
        delta_raw = self.head_corr(h_lstm)
        delta = self.correction_cap * torch.tanh(delta_raw)

        Q_pred = Q_phys * (1.0 + delta)
        Q_pred = torch.relu(Q_pred)

        heating_factor = torch.sigmoid(2.0 * (15.0 - T_out))
        Q_pred = Q_pred * heating_factor

        Q_pred_scaled = (Q_pred - y_mean) / y_std

        return {
            "Q_pred_scaled": Q_pred_scaled,
            "Q_pred_kW": Q_pred,
            "Q_phys_kW": Q_phys,
            "delta": delta,
            "k_env": k_env,
            "k_int": k_int,
            "k_sol": k_sol,
            "k_vent": k_vent,
        }

print("✅ Model classes defined")

# ================================================================
# 8. LOSS FUNCTION
# ================================================================
def compute_losses_v2(batch, out):
    y_true = batch["y"].to(device)
    y_pred = out["Q_pred_scaled"]
    
    y_true_kW = y_true * y_std + y_mean
    weights = 1.0 + 3.0 * torch.relu((y_true_kW - 100) / 150).clamp(0, 1)
    
    msum = weights.sum().clamp_min(1.0)
    L_data = ((y_pred - y_true)**2 * weights).sum() / msum
    L_delta = (out["delta"]**2).mean()
    L_kreg = ((out["k_env"] - 1.0).pow(2).mean() + 
              (out["k_int"] - 1.0).pow(2).mean() + 
              (out["k_sol"] - 1.0).pow(2).mean() + 
              (out["k_vent"] - 1.0).pow(2).mean()) / 4.0

    L_total = L_data + LAMBDA_DELTA * L_delta + LAMBDA_KREG * L_kreg

    return L_total

# ================================================================
# 9. TRAINING FUNCTION
# ================================================================
def train_model(model, train_loader, val_loader, epochs=40, lr=3e-4, patience=6):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = compute_losses_v2(batch, out)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch)
                loss = compute_losses_v2(batch, out)
                val_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            marker = "✓"
        else:
            patience_counter += 1
            marker = ""
        
        if (epoch + 1) % 5 == 0 or patience_counter == 0:
            print(f"  Epoch {epoch+1:3d}: train={train_loss:.4f}, val={val_loss:.4f} {marker}")
        
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(best_state)
    return model

# ================================================================
# 10. EVALUATION FUNCTION
# ================================================================
def evaluate_model(model, loader):
    model.eval()
    all_pred, all_true = [], []
    
    with torch.no_grad():
        for batch in loader:
            out = model(batch)
            pred = out["Q_pred_kW"].cpu().numpy().flatten()
            y_scaled = batch["y"].numpy().flatten()
            true = y_scaled * y_std.cpu().numpy() + y_mean.cpu().numpy()
            
            all_pred.extend(pred)
            all_true.extend(true)
    
    all_pred = np.array(all_pred)
    all_true = np.array(all_true)
    
    rmse = np.sqrt(np.mean((all_pred - all_true)**2))
    mae = np.mean(np.abs(all_pred - all_true))
    ss_res = np.sum((all_true - all_pred)**2)
    ss_tot = np.sum((all_true - all_true.mean())**2)
    r2 = 1 - ss_res / ss_tot
    
    print(f"   RMSE: {rmse:.2f} kW | MAE: {mae:.2f} kW | R²: {r2:.4f}")
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2}

# ================================================================
# 11. TRAIN BASELINE MODEL
# ================================================================
print("\n" + "="*70)
print("STEP 7: TRAINING BASELINE MODEL")
print("="*70)

model = LSTMPINNHeatingV2_2_Enhanced(
    ts_dim=n_ts,
    static_dim=n_static,
    hidden_lstm=64,
    num_layers=1,
    bidirectional=False,
    dropout=0.0,
    setpoint=22.0,
    correction_cap=5.0,
    exposure_factor=20.0,
    ua_multiplier=1.2,
    schedule_weight=1.0
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {n_params:,}")

trained_model = train_model(model, train_loader, val_loader, epochs=40, lr=3e-4, patience=6)

print(f"\n📊 TEST PERFORMANCE:")
metrics = evaluate_model(trained_model, test_loader)

# ================================================================
# 12. SAVE MODEL
# ================================================================
print("\n" + "="*70)
print("STEP 8: SAVING MODEL")
print("="*70)

config = {
    'name': 'Baseline',
    'hidden_lstm': 64,
    'num_layers': 1,
    'bidirectional': False,
    'dropout': 0.0
}

torch.save({
    "state_dict": trained_model.state_dict(),
    "config": config
}, "lstm_pgnn_v2_2_baseline.pt")

print(f"✅ Saved: lstm_pgnn_v2_2_baseline.pt")
print(f"\n🏆 BASELINE MODEL COMPLETE")
print(f"   R²={metrics['r2']:.4f} | RMSE={metrics['rmse']:.2f} kW | MAE={metrics['mae']:.2f} kW")