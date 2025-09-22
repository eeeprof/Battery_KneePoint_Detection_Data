#!/usr/bin/env python3
"""
Integrated Thermal and Efficiency Diagnostics for Lithium-Ion Cells
Robust analysis pipeline for prognostic assessment of battery degradation
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

# ======================= Configuration =======================
@dataclass
class AnalysisConfig:
    """Configuration parameters for battery analysis"""
    temp_threshold: float = 40.0      # °C
    roll_window: int = 7              # cycles for rolling variance
    knee_penalty: float = 5.0         # penalty in change-point objective
    r0_min_step_ma: float = 50.0      # mA step for current "jump"
    r0_window_points: int = 25        # points around step for R0 regression
    r0_window_seconds: float = 2.0    # time window for R0 estimation
    min_chg_mah: float = 10.0         # mAh minimum charge (reduced for validation)
    min_chg_wh: float = 0.05          # Wh minimum charge energy
    min_dis_mah: float = 10.0         # mAh minimum discharge
    min_dis_wh: float = 0.05          # Wh minimum discharge energy
    min_cycle_duration: float = 60.0  # seconds minimum cycle duration
    min_segment_length: int = 5       # minimum points for knee detection
    eff_clip: Tuple[float, float] = (0.0, 1.0)  # efficiency bounds
    max_expected_capacity_ah: float = 10.0  # Maximum expected capacity in Ah

    @property
    def required_columns(self) -> List[str]:
        return [
            "time_s", "Ecell_V", "I_mA", "EnergyCharge_W_h",
            "QCharge_mA_h", "EnergyDischarge_W_h", "QDischarge_mA_h",
            "Temperature__C", "cycleNumber"
        ]

# Global configuration
CONFIG = AnalysisConfig()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('battery_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Plot styling
plt.style.use('default')
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 30,
    "figure.figsize": (16, 8),
    "figure.dpi": 300,
    "savefig.dpi": 800,
    "savefig.bbox": "tight",
    "savefig.format": "png"
})

# ======================= Helper Functions =======================
def clip_efficiency(x: float, clip_to: Tuple[float, float] = CONFIG.eff_clip) -> float:
    """Clip efficiency values to valid range"""
    if pd.isna(x):
        return np.nan
    return float(np.clip(x, clip_to[0], clip_to[1]))

def safe_ratio(numerator: float, denominator: float) -> float:
    """Safe division with NaN handling"""
    if pd.isna(numerator) or pd.isna(denominator) or denominator <= 0:
        return np.nan
    return numerator / denominator

def has_column(df: pd.DataFrame, column: str) -> bool:
    """Check if column exists and has non-null values"""
    return (column in df.columns) and df[column].notna().any()

def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """Validate dataframe structure and data quality"""
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return False

    # Check for basic data quality
    if df.empty:
        logger.error("DataFrame is empty")
        return False

    if df['cycleNumber'].nunique() < 2:
        logger.warning("Fewer than 2 cycles detected")

    return True

def validate_capacity_values(capacity_mah: float, parameter_name: str) -> float:
    """Validate capacity values and convert to reasonable range"""
    if pd.isna(capacity_mah):
        return np.nan
    
    # Convert to Ah for validation
    capacity_ah = capacity_mah / 1000.0
    
    # Check if value is within reasonable bounds
    if capacity_ah > CONFIG.max_expected_capacity_ah:
        logger.warning(f"Unrealistic {parameter_name}: {capacity_ah:.3f} Ah. Checking data quality.")
        # Try to identify if there's a unit issue
        if capacity_ah > 1000:  # Definitely wrong - likely cumulative instead of cycle value
            return np.nan
    
    return capacity_mah

# ======================= Data Loading =======================
def load_battery_data(path: str) -> Optional[pd.DataFrame]:
    """Load and validate battery data from Excel file"""
    try:
        logger.info(f"Loading data from {path}")
        df = pd.read_excel(path)

        # Basic validation
        if not validate_dataframe(df, CONFIG.required_columns):
            return None

        # Sort and clean data
        df = df.sort_values(["cycleNumber", "time_s"]).reset_index(drop=True)

        # Remove duplicates
        initial_size = len(df)
        df = df.drop_duplicates(subset=["cycleNumber", "time_s"])
        if len(df) < initial_size:
            logger.warning(f"Removed {initial_size - len(df)} duplicate entries")

        # Validate capacity columns
        capacity_columns = ["QCharge_mA_h", "QDischarge_mA_h"]
        for col in capacity_columns:
            if col in df.columns:
                max_val = df[col].max()
                if max_val > CONFIG.max_expected_capacity_ah * 1000 * 10:  # 10x margin
                    logger.warning(f"Potential unit issue in {col}: max value = {max_val} mAh")

        logger.info(f"Successfully loaded {len(df)} records with {df['cycleNumber'].nunique()} cycles")
        return df

    except Exception as e:
        logger.error(f"Error loading data from {path}: {str(e)}")
        return None

# ======================= Charge/Discharge Segmentation =======================
def segment_charge_discharge(cycle_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Identify charge and discharge phases using multiple indicators
    Returns: charge_mask, discharge_mask, time, current, voltage
    """
    I = cycle_data["I_mA"].to_numpy(dtype=float)
    V = cycle_data["Ecell_V"].to_numpy(dtype=float)
    t = cycle_data["time_s"].to_numpy(dtype=float)

    # Primary segmentation based on current sign
    charge_mask = I > 1.0  # Small threshold to avoid noise around zero
    discharge_mask = I < -1.0

    # Secondary validation using cumulative counters (handle carefully)
    if has_column(cycle_data, "QCharge_mA_h"):
        dQchg = np.diff(cycle_data["QCharge_mA_h"].to_numpy(), prepend=0)
        charge_mask = charge_mask | (dQchg > 0.1)  # Small threshold

    if has_column(cycle_data, "EnergyCharge_W_h"):
        dEchg = np.diff(cycle_data["EnergyCharge_W_h"].to_numpy(), prepend=0)
        charge_mask = charge_mask | (dEchg > 0.001)  # Small threshold

    if has_column(cycle_data, "QDischarge_mA_h"):
        dQdis = np.diff(cycle_data["QDischarge_mA_h"].to_numpy(), prepend=0)
        discharge_mask = discharge_mask | (dQdis > 0.1)

    if has_column(cycle_data, "EnergyDischarge_W_h"):
        dEdis = np.diff(cycle_data["EnergyDischarge_W_h"].to_numpy(), prepend=0)
        discharge_mask = discharge_mask | (dEdis > 0.001)

    return charge_mask, discharge_mask, t, I, V

def integrate_phase(
    mask: np.ndarray,
    time_s: np.ndarray,
    current_ma: np.ndarray,
    voltage: Optional[np.ndarray] = None,
    phase_type: str = "charge"
) -> Tuple[float, float]:
    """
    Integrate capacity and energy over a phase using trapezoidal integration.
    Returns: (capacity_mAh, energy_Wh)
    """
    if np.sum(mask) < 2:
        return np.nan, np.nan

    # Use trapezoidal integration for better accuracy
    t_sel = time_s[mask]
    I_sel = current_ma[mask]
    
    # Calculate time differences in hours
    dt_h = np.diff(t_sel) / 3600.0  # hours
    
    # Use average current between points for integration
    I_avg = (I_sel[:-1] + I_sel[1:]) / 2.0
    
    if phase_type == "charge":
        # For charge, use positive current
        cap_mAh = np.sum(np.maximum(I_avg, 0) * dt_h)
    else:
        # For discharge, use absolute value of negative current
        cap_mAh = np.sum(np.abs(np.minimum(I_avg, 0)) * dt_h)

    # Energy integration
    if voltage is not None:
        V_sel = voltage[mask]
        V_avg = (V_sel[:-1] + V_sel[1:]) / 2.0
        
        if phase_type == "charge":
            power_W = V_avg * np.maximum(I_avg, 0) / 1000.0  # mW to W
        else:
            power_W = V_avg * np.abs(np.minimum(I_avg, 0)) / 1000.0
        
        energy_Wh = np.sum(power_W * dt_h)
    else:
        energy_Wh = np.nan

    # Validate results
    cap_mAh = validate_capacity_values(cap_mAh, f"{phase_type} capacity")
    
    return float(cap_mAh), float(energy_Wh)

def compute_cycle_energies(cycle_data: pd.DataFrame) -> Tuple[float, float, float, float]:
    """
    Compute charge and discharge capacities and energies for a single cycle
    Returns: Qchg (mAh), Echg (Wh), Qdis (mAh), Edis (Wh)
    """
    charge_mask, discharge_mask, time, current, voltage = segment_charge_discharge(cycle_data)

    # Integrate using measured signals (primary method)
    Qchg_raw, Echg_raw = integrate_phase(charge_mask, time, current, voltage, "charge")
    Qdis_raw, Edis_raw = integrate_phase(discharge_mask, time, current, voltage, "discharge")

    # Use counter-based values only if integration seems problematic
    if (pd.isna(Qchg_raw) or Qchg_raw > CONFIG.max_expected_capacity_ah * 1000 * 5) and has_column(cycle_data, "QCharge_mA_h"):
        Qchg_counter = cycle_data["QCharge_mA_h"].max() - cycle_data["QCharge_mA_h"].min()
        Qchg_raw = validate_capacity_values(Qchg_counter, "charge capacity from counter")

    if (pd.isna(Echg_raw) or Echg_raw > 1000) and has_column(cycle_data, "EnergyCharge_W_h"):
        Echg_counter = cycle_data["EnergyCharge_W_h"].max() - cycle_data["EnergyCharge_W_h"].min()
        Echg_raw = Echg_counter if Echg_counter < 100 else np.nan  # Sanity check

    if (pd.isna(Qdis_raw) or Qdis_raw > CONFIG.max_expected_capacity_ah * 1000 * 5) and has_column(cycle_data, "QDischarge_mA_h"):
        Qdis_counter = cycle_data["QDischarge_mA_h"].max() - cycle_data["QDischarge_mA_h"].min()
        Qdis_raw = validate_capacity_values(Qdis_counter, "discharge capacity from counter")

    if (pd.isna(Edis_raw) or Edis_raw > 1000) and has_column(cycle_data, "EnergyDischarge_W_h"):
        Edis_counter = cycle_data["EnergyDischarge_W_h"].max() - cycle_data["EnergyDischarge_W_h"].min()
        Edis_raw = Edis_counter if Edis_counter < 100 else np.nan

    # Final validation
    Qchg = Qchg_raw if not pd.isna(Qchg_raw) and Qchg_raw >= 0 else np.nan
    Echg = Echg_raw if not pd.isna(Echg_raw) and Echg_raw >= 0 else np.nan
    Qdis = Qdis_raw if not pd.isna(Qdis_raw) and Qdis_raw >= 0 else np.nan
    Edis = Edis_raw if not pd.isna(Edis_raw) and Edis_raw >= 0 else np.nan

    return Qchg, Echg, Qdis, Edis

# ======================= Per-Cycle Analysis =======================
def analyze_single_cycle(cycle_data: pd.DataFrame) -> Dict:
    """Compute comprehensive statistics for a single cycle"""
    cycle_number = int(cycle_data["cycleNumber"].iloc[0])

    # Compute energies and capacities (mAh, Wh)
    Qchg, Echg, Qdis, Edis = compute_cycle_energies(cycle_data)

    # Calculate efficiencies with validation
    if (not pd.isna(Qchg) and not pd.isna(Qdis) and
        Qchg >= CONFIG.min_chg_mah and Qdis >= CONFIG.min_dis_mah and
        Qchg < CONFIG.max_expected_capacity_ah * 1000 and 
        Qdis < CONFIG.max_expected_capacity_ah * 1000):
        eta_Q = clip_efficiency(safe_ratio(Qdis, Qchg))
    else:
        eta_Q = np.nan

    if (not pd.isna(Echg) and not pd.isna(Edis) and
        Echg >= CONFIG.min_chg_wh and Edis >= CONFIG.min_dis_wh and
        Echg < 1000 and Edis < 1000):  # Reasonable energy bounds
        eta_E = clip_efficiency(safe_ratio(Edis, Echg))
    else:
        eta_E = np.nan

    # Temperature statistics
    T_mean = float(cycle_data["Temperature__C"].mean())
    T_95 = float(np.percentile(cycle_data["Temperature__C"], 95))

    # Time fraction above temperature threshold
    time_values = cycle_data["time_s"].to_numpy(dtype=float)
    if len(time_values) > 1:
        dt = np.diff(time_values, prepend=time_values[0])
        hot_mask = (cycle_data["Temperature__C"].to_numpy(dtype=float) > CONFIG.temp_threshold)
        total_time = float(np.sum(dt))
        hot_time = float(np.sum(dt * hot_mask))
        frac_hot = hot_time / total_time if total_time > 0 else 0.0
    else:
        frac_hot = 0.0

    # Cycle duration
    cycle_duration = float(time_values[-1] - time_values[0]) if len(time_values) > 1 else 0.0

    return {
        "cycleNumber": cycle_number,
        "Qchg_mAh": Qchg, "Echg_Wh": Echg, "Qdis_mAh": Qdis, "Edis_Wh": Edis,
        "eta_Q": eta_Q, "eta_E": eta_E,
        "T_mean": T_mean, "T_95": T_95, "frac_time_Tgt40": frac_hot,
        "cycle_duration": cycle_duration
    }

def per_cycle_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate comprehensive summary statistics for all cycles"""
    rows = []

    for cycle_num, cycle_data in df.groupby("cycleNumber"):
        if len(cycle_data) < 10:  # Skip very short cycles
            logger.warning(f"Skipping cycle {cycle_num} with only {len(cycle_data)} data points")
            continue

        cycle_stats = analyze_single_cycle(cycle_data)
        rows.append(cycle_stats)

    result_df = pd.DataFrame(rows).sort_values("cycleNumber").reset_index(drop=True)
    
    # Log capacity statistics for validation
    valid_qdis = result_df["Qdis_mAh"].dropna()
    if len(valid_qdis) > 0:
        max_cap = valid_qdis.max()
        min_cap = valid_qdis.min()
        logger.info(f"Capacity range: {min_cap:.1f} - {max_cap:.1f} mAh ({min_cap/1000:.3f} - {max_cap/1000:.3f} Ah)")
    
    logger.info(f"Processed {len(result_df)} cycles")
    return result_df

# ======================= Internal Resistance Estimation =======================
def estimate_r0_single_cycle(cycle_data: pd.DataFrame) -> float:
    """Estimate internal resistance for a single cycle"""
    I = cycle_data["I_mA"].to_numpy(dtype=float) / 1000.0  # Convert to A
    V = cycle_data["Ecell_V"].to_numpy(dtype=float)
    time = cycle_data["time_s"].to_numpy(dtype=float)

    if len(I) < CONFIG.r0_window_points + 2:
        return np.nan

    # Find significant current changes using gradient
    dI_dt = np.gradient(I, time)
    significant_steps = np.where(np.abs(dI_dt) > (CONFIG.r0_min_step_ma / 1000.0) / 10.0)[0]

    r0_estimates = []

    for step_idx in significant_steps:
        # Create time-based window around the step
        t0 = time[step_idx]
        time_window = (time >= t0 - CONFIG.r0_window_seconds / 2) & \
                      (time <= t0 + CONFIG.r0_window_seconds / 2)

        if np.sum(time_window) < 5:
            continue

        I_window = I[time_window]
        V_window = V[time_window]

        # Remove outliers using IQR on current
        Q1 = np.percentile(I_window, 25)
        Q3 = np.percentile(I_window, 75)
        IQR = Q3 - Q1
        if IQR > 0:
            valid_mask = (I_window >= Q1 - 1.5 * IQR) & (I_window <= Q3 + 1.5 * IQR)
        else:
            valid_mask = np.ones_like(I_window, dtype=bool)

        if np.sum(valid_mask) >= 5:
            try:
                slope, intercept, r_value, p_value, std_err = linregress(
                    I_window[valid_mask], V_window[valid_mask]
                )
                # Validate the regression result
                if (0 < slope < 1.0 and r_value**2 > 0.8 and
                    not np.isnan(slope) and not np.isinf(slope)):
                    r0_estimates.append(slope)
            except Exception:
                continue

    return float(np.median(r0_estimates)) if r0_estimates else np.nan

def estimate_r0_all_cycles(df: pd.DataFrame) -> pd.Series:
    """Estimate internal resistance for all cycles"""
    r0_results = {}

    for cycle_num, cycle_data in df.groupby("cycleNumber"):
        if len(cycle_data) < 20:  # Skip cycles with insufficient data
            r0_results[int(cycle_num)] = np.nan
            continue

        r0 = estimate_r0_single_cycle(cycle_data)
        r0_results[int(cycle_num)] = r0

    return pd.Series(r0_results, name="R0_ohm")

# ======================= Knee Point Detection =======================
def detect_knee_point(y: np.ndarray, min_segment_length: int = CONFIG.min_segment_length,
                      penalty: float = CONFIG.knee_penalty) -> Optional[int]:
    """
    Detect knee point using piecewise linear regression
    Returns: index of knee point or None if not detected
    """
    y = np.asarray(y, dtype=float)
    n = len(y)

    if n < 2 * min_segment_length:
        return None

    best_knee = None
    best_score = float('inf')
    x = np.arange(n)

    for potential_knee in range(min_segment_length, n - min_segment_length):
        # Left segment
        x_left = x[:potential_knee]
        y_left = y[:potential_knee]

        # Right segment
        x_right = x[potential_knee:]
        y_right = y[potential_knee:]

        try:
            # Fit left segment
            left_coeffs = np.polyfit(x_left, y_left, 1)
            left_pred = np.polyval(left_coeffs, x_left)
            left_residual = np.sum((y_left - left_pred) ** 2)

            # Fit right segment
            right_coeffs = np.polyfit(x_right, y_right, 1)
            right_pred = np.polyval(right_coeffs, x_right)
            right_residual = np.sum((y_right - right_pred) ** 2)

            # Total score with slope difference penalty
            total_score = (left_residual + right_residual +
                           penalty * np.abs(left_coeffs[0] - right_coeffs[0]))

            if total_score < best_score:
                best_score = total_score
                best_knee = potential_knee

        except Exception:
            continue

    return best_knee

# ======================= Visualization =======================
def create_energy_efficiency_plot(cycle_data: pd.DataFrame, knee_cycle: Optional[int], domain: str, output_dir: str = "."):
    """Create energy efficiency plot with knee point annotation"""
    plt.figure(figsize=(16, 8))
    
    plt.plot(cycle_data["cycleNumber"], cycle_data["eta_E"], 'o-', markersize=6, 
             linewidth=2, label=r"Energy efficiency $\eta_E$")
    
    if knee_cycle is not None:
        plt.axvline(knee_cycle, color='red', linestyle="--", linewidth=3,
                    label=f"Knee point @ cycle {int(knee_cycle)}")
    
    plt.xlabel("Cycle Number", fontsize=30)
    plt.ylabel(r"Energy Efficiency $\eta_E$", fontsize=30)
    plt.title(f"{domain} - Energy Efficiency", fontsize=30, fontweight='bold')
    plt.legend(fontsize=30)
    plt.grid(True, alpha=0.3)
    plt.ylim(0.8, 1.02)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{domain}_energy_efficiency.png"), dpi=300)
    plt.close()

def create_charge_efficiency_plot(cycle_data: pd.DataFrame, knee_cycle: Optional[int], domain: str, output_dir: str = "."):
    """Create charge efficiency plot with variance"""
    plt.figure(figsize=(16, 8))
    
    # Primary axis for efficiency
    plt.plot(cycle_data["cycleNumber"], cycle_data["eta_Q"], 'o-', markersize=6, 
             linewidth=2, label=r"Charge efficiency $\eta_Q$", color='blue')
    
    # Secondary axis for variance
    ax2 = plt.gca().twinx()
    var_etaQ = cycle_data["eta_Q"].rolling(CONFIG.roll_window, min_periods=1, center=True).var()
    ax2.plot(cycle_data["cycleNumber"], var_etaQ, '--', linewidth=2,
             label=f"Variance (window={CONFIG.roll_window})", color='orange')
    ax2.set_ylabel("Efficiency Variance", fontsize=30, color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    
    if knee_cycle is not None:
        plt.axvline(knee_cycle, color='red', linestyle="--", linewidth=3,
                    label=f"Knee point @ cycle {int(knee_cycle)}")
    
    plt.xlabel("Cycle Number", fontsize=30)
    plt.ylabel(r"Charge Efficiency $\eta_Q$", fontsize=30, color='blue')
    plt.title(f"{domain} - Charge Efficiency and Variance", fontsize=30, fontweight='bold')
    
    # Combine legends
    lines1, labels1 = plt.gca().get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.gca().legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=30)
    
    plt.grid(True, alpha=0.3)
    plt.ylim(0.8, 1.02)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{domain}_charge_efficiency.png"), dpi=300)
    plt.close()

def create_temperature_plot(cycle_data: pd.DataFrame, knee_cycle: Optional[int], domain: str, output_dir: str = "."):
    """Create temperature statistics plot"""
    plt.figure(figsize=(16, 8))

    plt.plot(cycle_data["cycleNumber"], cycle_data["T_mean"], 'o-', markersize=6, linewidth=2,
             label="Mean Temperature (°C)")
    plt.plot(cycle_data["cycleNumber"], cycle_data["T_95"], 's-', markersize=4, linewidth=2,
             label="95th Percentile Temperature (°C)")
    plt.plot(cycle_data["cycleNumber"], cycle_data["frac_time_Tgt40"] * 100, '^-', markersize=5, linewidth=2,
             label=f"Time > {CONFIG.temp_threshold}°C (%)")

    if knee_cycle is not None:
        plt.axvline(knee_cycle, color='red', linestyle="--", linewidth=3,
                    label=f"Knee point @ cycle {int(knee_cycle)}")

    plt.xlabel("Cycle Number", fontsize=30)
    plt.ylabel("Temperature (°C) / Time Fraction (%)", fontsize=30)
    plt.title(f"{domain} - Temperature Statistics", fontsize=30, fontweight='bold')
    plt.legend(fontsize=30)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{domain}_temperature_stats.png"), dpi=600)
    plt.close()

def create_resistance_plot(cycle_data: pd.DataFrame, knee_cycle: Optional[int], domain: str, output_dir: str = "."):
    """Create internal resistance plot with piecewise fits"""
    plt.figure(figsize=(16, 8))

    plt.plot(cycle_data["cycleNumber"], cycle_data["R0_ohm"], 'o-', markersize=6, linewidth=2,
             label=r"Internal Resistance $R_0$ ($\Omega$)")

    if knee_cycle is not None and not pd.isna(knee_cycle):
        knee_idx = cycle_data[cycle_data["cycleNumber"] == knee_cycle].index
        if len(knee_idx) > 0:
            knee_idx = knee_idx[0]

            # Fit before knee
            before_data = cycle_data.iloc[:knee_idx+1]
            if len(before_data) > 2 and before_data["R0_ohm"].notna().sum() > 1:
                x_before = before_data["cycleNumber"]
                y_before = before_data["R0_ohm"]
                valid_mask = y_before.notna()
                if valid_mask.sum() > 1:
                    slope_b, intercept_b, *_ = linregress(x_before[valid_mask], y_before[valid_mask])
                    plt.plot(x_before, intercept_b + slope_b * x_before, '-', linewidth=2,
                             label=f"Pre-knee slope: {slope_b:.2e} Ω/cycle")

            # Fit after knee
            after_data = cycle_data.iloc[knee_idx:]
            if len(after_data) > 2 and after_data["R0_ohm"].notna().sum() > 1:
                x_after = after_data["cycleNumber"]
                y_after = after_data["R0_ohm"]
                valid_mask = y_after.notna()
                if valid_mask.sum() > 1:
                    slope_a, intercept_a, *_ = linregress(x_after[valid_mask], y_after[valid_mask])
                    plt.plot(x_after, intercept_a + slope_a * x_after, '-', linewidth=2,
                             label=f"Post-knee slope: {slope_a:.2e} Ω/cycle")

    if knee_cycle is not None:
        plt.axvline(knee_cycle, color='red', linestyle="--", linewidth=3,
                    label=f"Knee point @ cycle {int(knee_cycle)}")

    plt.xlabel("Cycle Number", fontsize=30)
    plt.ylabel(r"Internal Resistance $R_0$ ($\Omega$)", fontsize=30)
    plt.title(f"{domain} - Internal Resistance Evolution", fontsize=30, fontweight='bold')
    plt.legend(fontsize=30)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{domain}_internal_resistance.png"), dpi=300)
    plt.close()

def create_capacity_plot(cycle_data: pd.DataFrame, knee_cycle: Optional[int], domain: str, output_dir: str = "."):
    """Create capacity degradation plot"""
    plt.figure(figsize=(16, 8))

    # Plot in Ah for better readability
    Qdis_Ah = cycle_data["Qdis_mAh"] / 1000.0
    Qchg_Ah = cycle_data["Qchg_mAh"] / 1000.0
    
    plt.plot(cycle_data["cycleNumber"], Qdis_Ah, 'o-', markersize=6, linewidth=2,
             label="Discharge Capacity (Ah)")
    plt.plot(cycle_data["cycleNumber"], Qchg_Ah, 's-', markersize=4, linewidth=2,
             label="Charge Capacity (Ah)")

    if knee_cycle is not None:
        plt.axvline(knee_cycle, color='red', linestyle="--", linewidth=3,
                    label=f"Knee point @ cycle {int(knee_cycle)}")

    plt.xlabel("Cycle Number", fontsize=30)
    plt.ylabel("Capacity (Ah)", fontsize=30)
    plt.title(f"{domain} - Capacity Degradation", fontsize=30, fontweight='bold')
    plt.legend(fontsize=30)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{domain}_capacity_degradation.png"), dpi=300)
    plt.close()

def create_combined_efficiency_plot(cycle_data: pd.DataFrame, knee_cycle: Optional[int], domain: str, output_dir: str = "."):
    """Create combined efficiency plot"""
    plt.figure(figsize=(16, 8))
    
    plt.plot(cycle_data["cycleNumber"], cycle_data["eta_E"], 'o-', markersize=6, linewidth=2,
             label=r"Energy efficiency $\eta_E$")
    plt.plot(cycle_data["cycleNumber"], cycle_data["eta_Q"], 's-', markersize=4, linewidth=2,
             label=r"Charge efficiency $\eta_Q$")
    
    if knee_cycle is not None:
        plt.axvline(knee_cycle, color='red', linestyle="--", linewidth=3,
                    label=f"Knee point @ cycle {int(knee_cycle)}")
    
    plt.xlabel("Cycle Number", fontsize=30)
    plt.ylabel("Efficiency", fontsize=30)
    plt.title(f"{domain} - Combined Efficiency Metrics", fontsize=30, fontweight='bold')
    plt.legend(fontsize=30)
    plt.grid(True, alpha=0.3)
    plt.ylim(0.8, 1.02)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{domain}_combined_efficiency.png"), dpi=300)
    plt.close()

# ======================= Main Analysis Pipeline =======================
def analyze_battery_data(file_paths: List[str], output_dir: str = "results") -> Dict[str, pd.DataFrame]:
    """
    Main analysis pipeline for battery data
    Returns: Dictionary of results for each domain
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    results: Dict[str, pd.DataFrame] = {}
    knee_points: Dict[str, Optional[int]] = {}

    logger.info(f"Starting analysis of {len(file_paths)} files")

    for i, file_path in enumerate(file_paths):
        domain = os.path.basename(file_path).split(".")[0]
        logger.info(f"Processing {i+1}/{len(file_paths)}: {domain}")

        try:
            # Load and preprocess data
            df = load_battery_data(file_path)
            if df is None:
                continue

            # Generate cycle summary
            cycle_summary = per_cycle_summary(df)

            # Estimate internal resistance
            r0_estimates = estimate_r0_all_cycles(df)
            r0_df = r0_estimates.rename_axis('cycleNumber').reset_index()
            cycle_summary = cycle_summary.merge(r0_df, on="cycleNumber", how="left")

            # Detect knee points (using efficiency signatures)
            valid_cycles = cycle_summary.dropna(subset=["eta_E", "eta_Q"])
            if len(valid_cycles) >= 10:
                # Smooth energy efficiency for knee detection
                etaE_smoothed = valid_cycles["eta_E"].rolling(5, min_periods=1, center=True).mean().to_numpy()
                knee_etaE = detect_knee_point(etaE_smoothed)

                # Variance of charge efficiency
                etaQ_var = valid_cycles["eta_Q"].rolling(CONFIG.roll_window, min_periods=1, center=True).var().to_numpy()
                knee_var = detect_knee_point(etaQ_var)

                # Select the earliest consistent knee
                knees = [k for k in [knee_etaE, knee_var] if k is not None]
                if knees:
                    global_knee = int(valid_cycles["cycleNumber"].iloc[min(knees)])
                    knee_points[domain] = global_knee
                    logger.info(f"Detected knee point for {domain} at cycle {global_knee}")
                else:
                    knee_points[domain] = None
                    logger.warning(f"No knee point detected for {domain}")
            else:
                knee_points[domain] = None
                logger.warning(f"Insufficient data for knee detection in {domain}")

            # Store results
            cycle_summary["domain"] = domain
            results[domain] = cycle_summary

            # Generate individual plots
            knee_cycle = knee_points.get(domain)
            create_energy_efficiency_plot(cycle_summary, knee_cycle, domain, output_dir)
            create_charge_efficiency_plot(cycle_summary, knee_cycle, domain, output_dir)
            create_combined_efficiency_plot(cycle_summary, knee_cycle, domain, output_dir)
            create_temperature_plot(cycle_summary, knee_cycle, domain, output_dir)
            create_resistance_plot(cycle_summary, knee_cycle, domain, output_dir)
            create_capacity_plot(cycle_summary, knee_cycle, domain, output_dir)

            # Save results to CSV
            output_file = os.path.join(output_dir, f"{domain}_results.csv")
            cycle_summary.to_csv(output_file, index=False)
            logger.info(f"Saved results to {output_file}")

        except Exception as e:
            logger.error(f"Error processing {domain}: {str(e)}")
            continue

    # Generate summary report
    generate_summary_report(results, knee_points, output_dir)

    logger.info("Analysis completed successfully")
    return results

def generate_summary_report(results: Dict[str, pd.DataFrame], knee_points: Dict[str, Optional[int]],
                            output_dir: str):
    """Generate a comprehensive summary report"""
    report_lines = ["Battery Analysis Summary Report", "=" * 50, ""]

    for domain, data in results.items():
        report_lines.append(f"Domain: {domain}")
        report_lines.append(f"  Total cycles: {len(data)}")

        # Get valid capacity values
        valid_qdis = data['Qdis_mAh'].dropna()
        if len(valid_qdis) > 0:
            qdis_max = valid_qdis.max()
            qdis_min = valid_qdis.min()
            
            # Convert to Ah for reporting
            report_lines.append(f"  Initial capacity (max Qdis): {qdis_max:.1f} mAh ({qdis_max/1000.0:.3f} Ah)")
            report_lines.append(f"  Final capacity (min Qdis): {qdis_min:.1f} mAh ({qdis_min/1000.0:.3f} Ah)")
            
            if qdis_max > 0:
                retention = (qdis_min / qdis_max) * 100.0
                report_lines.append(f"  Capacity retention: {retention:.1f}%")
            else:
                report_lines.append("  Capacity retention: N/A (max capacity is zero)")
        else:
            report_lines.append("  Capacity data: No valid discharge capacity measurements")

        knee_cycle = knee_points.get(domain)
        if knee_cycle is not None:
            report_lines.append(f"  Detected knee point: Cycle {knee_cycle}")
        else:
            report_lines.append("  Knee point: Not detected")

        report_lines.append("")

    # Write report to file
    report_file = os.path.join(output_dir, "analysis_summary.txt")
    with open(report_file, 'w') as f:
        f.write("\n".join(report_lines))

    logger.info(f"Summary report saved to {report_file}")

# ======================= Main Execution =======================
if __name__ == "__main__":
    # Example usage — update these file names to your paths
    FILES = ["Cell1.xlsx", "Cell2.xlsx", "Cell3.xlsx"]

    # Run analysis
    results = analyze_battery_data(FILES, output_dir="battery_analysis_results")

    print("\nAnalysis completed. Check the 'battery_analysis_results' directory for outputs.")
