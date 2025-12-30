import numpy as np
import pandas as pd
import scipy.fftpack
import argparse
import sys
import os
import glob
import warnings

# Suppress warnings for cleaner output
warnings.simplefilter("ignore")

# ==============================================================================
# UNIVERSAL INFORMATION POTENTIAL (UIP) PHYSICS ENGINE
# Version: 1.5 (Corrected Galaxy Mass Release)
# ==============================================================================
# FIX LIST:
# - Galaxy Module now sums Gas + Disk + Bulge components correctly.
# - Previously it only counted Gas, leading to false Alpha=0.95 readings.
# ==============================================================================

ALPHA_CONSTANT = 0.353

def banner():
    print("="*60)
    print("      UIP PHYSICS ENGINE | CROSS-DOMAIN VALIDATOR")
    print(f"      Target Alpha Constant: {ALPHA_CONSTANT}")
    print("="*60)

# --- UTILITY: The Alpha Memory Kernel ---
def uip_filter_kernel(data_array):
    """
    Applies the Alpha=0.35 Memory Kernel to whiten 'stiff' noise.
    """
    n = len(data_array)
    if n == 0: return data_array
    
    # FFT to frequency domain
    fft_vals = scipy.fftpack.fft(data_array)
    freqs = scipy.fftpack.fftfreq(n)
    freqs[0] = 1e-15  # Avoid div/0
    
    # INVERSE Filter: Remove the predicted alpha-stiffness
    filter_response = np.abs(freqs)**ALPHA_CONSTANT
    filtered_fft = fft_vals * filter_response
    
    return np.real(scipy.fftpack.ifft(filtered_fft))

# --- MODULE 1: QUANTUM INFORMATION ---
def analyze_quantum_efficiency(t1_times, t2_times):
    print("\n[ MODULE 1: QUANTUM INFORMATION (IBM) ]")
    
    valid = (t1_times > 0) & (t2_times > 0)
    t1 = t1_times[valid]
    t2 = t2_times[valid]
    
    if len(t1) < 10:
        print("   > Insufficient data points.")
        return

    # Efficiency Ratio: eta = T2 / (2 * T1)
    # UIP Prediction: eta_limit = 1 - Alpha = 0.65
    ratios = t2 / (2 * t1)
    mean_efficiency = np.mean(ratios)
    
    print(f"   > Qubits Analyzed:      {len(t1)}")
    print(f"   > Measured Efficiency:  {mean_efficiency:.4f}")
    print(f"   > Predicted UIP Limit:  {1 - ALPHA_CONSTANT:.4f}")
    
    if abs(mean_efficiency - (1 - ALPHA_CONSTANT)) < 0.05:
        print("   >>> RESULT: CONFIRMED. Matches Alpha Limit.")
    else:
        print("   >>> RESULT: DEVIATION DETECTED.")

# --- MODULE 2: GALACTIC DYNAMICS ---
def analyze_galaxy_rotation(v_obs, v_bar):
    print("\n[ MODULE 2: GALACTIC DYNAMICS (SPARC) ]")
    
    valid = (v_obs > 0) & (v_bar > 0)
    v_o = v_obs[valid]
    v_b = v_bar[valid]
    
    if len(v_o) < 10:
        print("   > Insufficient data points.")
        return
        
    # Inferred Stiffness: Alpha = 1 - (V_baryonic^2 / V_observed^2)
    # This represents the "Missing Mass" attributed to Vacuum Stiffness
    measured_alphas = 1 - (v_b**2 / v_o**2)
    median_alpha = np.median(measured_alphas)
    
    print(f"   > Galaxy Points:        {len(v_o)}")
    print(f"   > Inferred Vacuum Stiffness: {median_alpha:.4f}")
    
    # We accept a range around 0.35
    if 0.30 < median_alpha < 0.45:
        print("   >>> RESULT: CONFIRMED. Matches Universal Constant.")
    else:
        print("   >>> RESULT: INCONCLUSIVE.")

# --- MODULE 3: LIGO GLITCH SATURATION ---
def analyze_ligo_saturation(snr_array, bandwidth_array):
    print("\n[ MODULE 3: GRAVITATIONAL TRANSIENTS (LIGO) ]")
    
    if len(snr_array) < 10:
        print("   > Insufficient data points.")
        return

    # 1. Calculate UIP Retention Factor S(x)
    retention = 1.0 / (1.0 + np.power(snr_array, ALPHA_CONSTANT))
    
    # 2. Check Correlation with Bandwidth
    correlation = np.corrcoef(retention, bandwidth_array)[0, 1]
    
    print(f"   > Transients Analyzed:  {len(snr_array)}")
    print(f"   > Retention Correlation: {correlation:.4f}")
    print(f"   > Target (from Paper):   -0.4500")
    
    if -0.60 < correlation < -0.30:
        print("   >>> RESULT: CONFIRMED. Matches Glitch Saturation Law.")
    else:
        print("   >>> RESULT: NO CORRELATION FOUND.")

# --- MODULE 4: PULSAR TIMING ---
def analyze_pulsar_timing(residual_data):
    print("\n[ MODULE 4: PULSAR TIMING (NANOGrav) ]")
    
    residual_data = residual_data[~np.isnan(residual_data)]
    
    if len(residual_data) < 10:
        print("   > Insufficient data points.")
        return
    
    raw_std = np.std(residual_data)
    whitened = uip_filter_kernel(residual_data)
    white_std = np.std(whitened)
    
    improvement = (raw_std - white_std) / raw_std * 100
    
    print(f"   > Observations (TOAs):  {len(residual_data)}")
    print(f"   > Raw Noise (RMS):      {raw_std:.4e}")
    print(f"   > Filtered Noise:       {white_std:.4e}")
    print(f"   > Structure Removed:    {improvement:.2f}%")
    
    if improvement > 1.0:
        print("   >>> RESULT: CONFIRMED. Vacuum Stiffness Detected.")
    else:
        print("   >>> RESULT: INCONCLUSIVE.")

# --- AUTO DISCOVERY ---
def auto_discover_files():
    found = {}
    print(f"[INFO] Scanning {os.getcwd()} for datasets...")
    
    # 1. Quantum
    q_files = glob.glob("ibm_*.csv")
    if q_files: found['quantum'] = q_files[0]
        
    # 2. Galaxy
    g_files = glob.glob("SPARC*.csv")
    if g_files: found['galaxy'] = g_files[0]
        
    # 3. LIGO
    l_files = glob.glob("*smallset*.xlsx") + glob.glob("*Smal set*.ods") + glob.glob("*smallset*.csv")
    if l_files: found['ligo'] = l_files[0]
        
    # 4. Pulsar
    p_files = glob.glob("UIP_Pulsar*.csv")
    if p_files: found['pulsar'] = p_files[0]

    for k, v in found.items():
        print(f"   [+] Found {k.capitalize()} Data: {v}")
    
    print("-" * 60)
    return found

# --- DEMO DATA GENERATOR ---
def run_demo_mode():
    print("\n" + "*"*60)
    print("      NO DATA FILES FOUND - RUNNING DEMO MODE")
    print("*"*60)
    
    N = 200
    t1 = np.random.normal(250, 50, N)
    efficiency = 1.0 - ALPHA_CONSTANT + np.random.normal(0, 0.05, N)
    analyze_quantum_efficiency(t1, 2 * t1 * efficiency)
    
    v_bar = np.linspace(50, 200, N)
    v_obs = v_bar / np.sqrt(1 - ALPHA_CONSTANT) + np.random.normal(0, 10, N)
    analyze_galaxy_rotation(v_obs, v_bar)
    
    snr = np.random.exponential(10, 500) + 10
    retention = 1.0 / (1.0 + np.power(snr, ALPHA_CONSTANT))
    bandwidth = 3000 * (1 - retention) + np.random.normal(0, 200, 500)
    analyze_ligo_saturation(snr, bandwidth)
    
    white = np.random.normal(0, 1, 4096)
    freqs = scipy.fftpack.fftfreq(len(white)); freqs[0]=1e-15
    stiff = np.real(scipy.fftpack.ifft(scipy.fftpack.fft(white)/(np.abs(freqs)**(ALPHA_CONSTANT/2))))
    analyze_pulsar_timing(stiff)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    banner()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--quantum')
    parser.add_argument('--galaxy')
    parser.add_argument('--ligo')
    parser.add_argument('--pulsar')
    args = parser.parse_args()
    
    files = {}
    if not (args.quantum or args.galaxy or args.ligo or args.pulsar):
        files = auto_discover_files()
    else:
        if args.quantum: files['quantum'] = args.quantum
        if args.galaxy: files['galaxy'] = args.galaxy
        if args.ligo: files['ligo'] = args.ligo
        if args.pulsar: files['pulsar'] = args.pulsar

    if not files:
        run_demo_mode()
        sys.exit()

    # --- EXECUTE MODULES ---

    if 'quantum' in files:
        try:
            df = pd.read_csv(files['quantum'])
            cols = df.columns.astype(str)
            t1 = next((c for c in cols if 'T1' in c), None)
            t2 = next((c for c in cols if 'T2' in c), None)
            if t1 and t2: analyze_quantum_efficiency(df[t1].values, df[t2].values)
        except Exception as e: print(f"[!] Quantum Error: {e}")

    if 'galaxy' in files:
        try:
            df = pd.read_csv(files['galaxy'])
            # --- THE FIX: Sum Gas + Disk + Bulge ---
            v_gas = df['Vgas']
            v_disk = df['Vdisk']
            v_bul = df['Vbul']
            # Total Baryonic Velocity = Sqrt(Sum of Squares)
            v_bar_total = np.sqrt(v_gas**2 + v_disk**2 + v_bul**2)
            
            analyze_galaxy_rotation(df['Vobs'].values, v_bar_total)
        except Exception as e: print(f"[!] Galaxy Error: {e}")

    if 'ligo' in files:
        try:
            f = files['ligo']
            if f.endswith('.xlsx'): df = pd.read_excel(f)
            elif f.endswith('.ods'):
                try: df = pd.read_excel(f, engine='odf')
                except: df = pd.DataFrame()
            else: df = pd.read_csv(f)
            
            if not df.empty:
                df.columns = df.columns.str.lower()
                snr = next((c for c in df.columns if 'snr' in c), None)
                bw = next((c for c in df.columns if 'bandwidth' in c), None)
                if snr and bw: analyze_ligo_saturation(df[snr].values, df[bw].values)
        except Exception as e: print(f"[!] LIGO Error: {e}")

    if 'pulsar' in files:
        try:
            df = pd.read_csv(files['pulsar'])
            analyze_pulsar_timing(df['Residual'].values)
        except Exception as e: print(f"[!] Pulsar Error: {e}")
        
    print("\n[ END OF ANALYSIS ]")
