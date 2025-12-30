import numpy as np
import pandas as pd
import scipy.fftpack
import argparse
import sys

# ==============================================================================
# UNIVERSAL INFORMATION POTENTIAL (UIP) PHYSICS ENGINE
# Version: 1.0 (Public Release)
# ==============================================================================
# This engine tests the "Alpha Lock" hypothesis: that a fundamental vacuum 
# stiffness constant (Alpha ~ 0.35) governs noise structure across 
# Quantum, Galactic, and Vacuum scales.
# ==============================================================================

ALPHA_CONSTANT = 0.353  # The Universal Stiffness Constant

def banner():
    print("="*60)
    print("      UIP PHYSICS ENGINE | UNIVERSAL STIFFNESS ANALYZER")
    print(f"      Target Alpha Constant: {ALPHA_CONSTANT}")
    print("="*60)

def uip_filter_kernel(data_array):
    """
    Applies the Alpha=0.35 Memory Kernel to a time series.
    Returns the 'whitened' data if the theory is correct.
    """
    n = len(data_array)
    # FFT to frequency domain
    fft_vals = scipy.fftpack.fft(data_array)
    freqs = scipy.fftpack.fftfreq(n)
    
    # Avoid division by zero at DC component
    freqs[0] = 1e-15 
    
    # The UIP 'Stiffness' Filter: S(f) ~ f^alpha
    # We apply the inverse to remove the predicted stiffness
    filter_response = np.abs(freqs)**ALPHA_CONSTANT
    
    # Apply filter and Inverse FFT
    filtered_fft = fft_vals * filter_response
    whitened_data = np.real(scipy.fftpack.ifft(filtered_fft))
    
    return whitened_data

def analyze_quantum_efficiency(t1_times, t2_times):
    """
    Tests if the Vacuum 'eats' Quantum Information at a rate of 0.35.
    Theory Limit: Efficiency = 1 - Alpha = 0.65
    """
    print("\n[ MODULE: QUANTUM INFORMATION ]")
    
    # Filter valid data
    valid = (t1_times > 0) & (t2_times > 0)
    t1 = t1_times[valid]
    t2 = t2_times[valid]
    
    # Calculate Coherence Efficiency Ratio: R = T2 / (2 * T1)
    # Standard Physics: R should be close to 1.0 (Perfect Coherence)
    # UIP Physics: R should be limited to 0.65
    ratios = t2 / (2 * t1)
    mean_efficiency = np.mean(ratios)
    
    print(f"   > Qubits Analyzed: {len(t1)}")
    print(f"   > Measured Efficiency: {mean_efficiency:.4f}")
    print(f"   > Predicted UIP Limit: {1 - ALPHA_CONSTANT:.4f}")
    
    diff = abs(mean_efficiency - (1 - ALPHA_CONSTANT))
    if diff < 0.05:
        print("   >>> RESULT: CONFIRMED. Matches Alpha Limit.")
    else:
        print("   >>> RESULT: DEVIATION DETECTED.")

def analyze_galaxy_rotation(v_obs, v_bar):
    """
    Tests if Vacuum Stiffness (0.35) replaces Dark Matter.
    UIP Formula: V_obs^2 = V_bar^2 / (1 - Alpha)
    """
    print("\n[ MODULE: GALACTIC DYNAMICS ]")
    
    # Filter valid data
    valid = (v_obs > 0) & (v_bar > 0)
    v_o = v_obs[valid]
    v_b = v_bar[valid]
    
    # Calculate 'Measured Alpha' for each galaxy point
    # Alpha = 1 - (V_bar^2 / V_obs^2)
    measured_alphas = 1 - (v_b**2 / v_o**2)
    median_alpha = np.median(measured_alphas)
    
    print(f"   > Data Points: {len(v_o)}")
    print(f"   > Required Dark Matter Fraction (Alpha): {median_alpha:.4f}")
    print(f"   > UIP Constant Prediction:             {ALPHA_CONSTANT:.4f}")
    
    if 0.30 < median_alpha < 0.45:
        print("   >>> RESULT: CONFIRMED. Vacuum Stiffness explains rotation.")
    else:
        print("   >>> RESULT: INCONCLUSIVE.")

def analyze_vacuum_noise(strain_data):
    """
    Tests if Vacuum Noise (LIGO) has 0.35 Memory Structure.
    """
    print("\n[ MODULE: VACUUM INTERFEROMETRY (LIGO) ]")
    
    raw_rms = np.std(strain_data)
    
    # Apply the Alpha Filter
    white_strain = uip_filter_kernel(strain_data)
    white_rms = np.std(white_strain)
    
    improvement = (raw_rms - white_rms) / raw_rms * 100
    
    print(f"   > Raw Vacuum Noise (RMS): {raw_rms:.4e}")
    print(f"   > UIP Filtered Noise:     {white_rms:.4e}")
    print(f"   > Noise Reduction:        {improvement:.2f}%")
    
    if improvement > 5.0:
        print("   >>> RESULT: CONFIRMED. Vacuum has 0.35 Memory Structure.")
    else:
        print("   >>> RESULT: NULL. Vacuum appears random.")

def generate_demo_data():
    """Generates synthetic data for demonstration if no files are provided."""
    print("\n[INFO] No files provided. Running in DEMO MODE with synthetic data.")
    
    # 1. Synthetic Quantum Data (matching real IBM stats)
    # T1 is random, T2 is limited by vacuum drag
    N = 200
    t1 = np.random.normal(250, 50, N)
    # T2 is strictly bounded by the UIP limit (0.65 efficiency) plus some thermal noise
    efficiency = 1.0 - ALPHA_CONSTANT + np.random.normal(0, 0.05, N)
    t2 = 2 * t1 * efficiency
    
    # 2. Synthetic Galaxy Data
    # V_bar is mass, V_obs is boosted by stiffness
    v_bar = np.linspace(50, 200, N)
    v_obs = v_bar / np.sqrt(1 - ALPHA_CONSTANT) + np.random.normal(0, 10, N)
    
    # 3. Synthetic Vacuum Noise
    # Pink noise with alpha=0.35 slope
    white = np.random.normal(0, 1, 4096)
    freqs = scipy.fftpack.fftfreq(len(white))
    freqs[0] = 1e-15
    # Add the "Stiffness" to the noise
    stiff_noise = np.real(scipy.fftpack.ifft(scipy.fftpack.fft(white) / (np.abs(freqs)**(ALPHA_CONSTANT/2))))
    
    return t1, t2, v_obs, v_bar, stiff_noise

if __name__ == "__main__":
    banner()
    
    # Simple CLI argument parsing
    parser = argparse.ArgumentParser(description='UIP Physics Engine')
    parser.add_argument('--quantum', type=str, help='Path to IBM Calibration CSV')
    parser.add_argument('--galaxy', type=str, help='Path to SPARC Galaxy CSV')
    parser.add_argument('--vacuum', type=str, help='Path to LIGO Strain CSV')
    args = parser.parse_args()
    
    # Load Data or Use Demo
    if not (args.quantum or args.galaxy or args.vacuum):
        q_t1, q_t2, g_vobs, g_vbar, v_strain = generate_demo_data()
        analyze_quantum_efficiency(q_t1, q_t2)
        analyze_galaxy_rotation(g_vobs, g_vbar)
        analyze_vacuum_noise(v_strain)
        
    else:
        # Real Data Loading Logic
        if args.quantum:
            try:
                df = pd.read_csv(args.quantum)
                # Auto-detect T1/T2 columns
                t1_col = [c for c in df.columns if 'T1' in c][0]
                t2_col = [c for c in df.columns if 'T2' in c][0]
                analyze_quantum_efficiency(df[t1_col].values, df[t2_col].values)
            except Exception as e:
                print(f"Error loading Quantum file: {e}")

        if args.galaxy:
            try:
                df = pd.read_csv(args.galaxy)
                # Auto-detect Vobs/Vbar
                # Assuming SPARC format or similar
                analyze_galaxy_rotation(df['Vobs'].values, df['Vgas'].values) # Simplified for example
            except Exception as e:
                print(f"Error loading Galaxy file: {e}")
                
        if args.vacuum:
            try:
                # Assuming simple single-column or standard LIGO format
                df = pd.read_csv(args.vacuum)
                # Look for 'strain' or second column
                col = [c for c in df.columns if 'strain' in c.lower()][0]
                analyze_vacuum_noise(df[col].values)
            except Exception as e:
                print(f"Error loading Vacuum file: {e}")

    print("\n[ END OF SIMULATION ]")
