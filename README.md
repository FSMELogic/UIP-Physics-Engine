# UIP Physics Engine: The Universal Stiffness Constant

**Version:** 1.0  
**License:** GPL3.0

## Overview
This repository contains the validation suite for the **Unified Information Potential (UIP) Theory**. 

The theory posits that spacetime is not an empty void, but a fluid-like medium with a fundamental **Vacuum Stiffness Constant** ($\alpha \approx 0.35$). This constant acts as a universal speed limit for information density, governing phenomena from the quantum scale to the galactic scale.

## The Evidence
This engine validates the constant against three distinct datasets:

1.  **Quantum Information (Micro):** * **Hypothesis:** The vacuum degrades quantum coherence (T2) relative to energy relaxation (T1).
    * **Prediction:** Efficiency Limit $\eta = 1 - \alpha \approx 0.65$.
    * **Result:** IBM Quantum Processors show a coherence ceiling of **0.64**.

2.  **Galactic Dynamics (Macro):**
    * **Hypothesis:** Vacuum stiffness exerts pressure on rotating galaxies, mimicking "Dark Matter."
    * **Prediction:** The missing mass fraction corresponds to $\alpha \approx 0.35$.
    * **Result:** SPARC Galaxy data confirms a stiffness coefficient of **0.37**.

3.  **Vacuum Noise (Fundamental):**
    * **Hypothesis:** Gravitational wave detector noise (LIGO) is not random shot noise, but has a memory structure.
    * **Prediction:** An $\alpha=0.35$ filter will whiten the noise floor.
    * **Result:** Noise reduction of **~15%** achieved on LIGO O3a data.
## Prediction
UIP predicts that the noise floor of the future LISA gravitational wave observatory will exhibit a 'Pink Noise' power spectrum with a slope of S(f) \propto f^{-0.35}.

## Usage

### 1. Run the Demo
If you don't have datasets, run the script in Demo Mode to see the math in action on synthetic data:
```bash
python uip_analyzer.py

