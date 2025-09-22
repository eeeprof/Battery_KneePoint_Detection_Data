# Battery_KneePoint_Detection_Data
Battery_KneePoint_Detection_Data provides datasets and analysis scripts for lithium-ion cell degradation studies. It includes cycle-wise efficiency, thermal, and resistance metrics with knee point detection results, supporting state-of-health (SoH) monitoring and remaining useful life (RUL) forecasting.


- **data/** → contains raw experimental datasets for Cell1, Cell2, and Cell3.  
- **scripts/** → Python scripts for preprocessing, analysis, efficiency calculation, resistance estimation, thermal metrics, and knee point detection.  
- **results/** → processed figures, cycle statistics, and final tables used in the paper.  
- **requirements.txt** → list of Python dependencies.  
- **LICENSE** → open-source license (MIT recommended).  

---

## Key Features
1. **Cycle-wise statistical aggregation**  
   - Extraction of maximum charge/discharge capacity and energy.  
   - Calculation of mean and 95th percentile temperature per cycle.  
   - Fraction of time spent above 40 °C as a thermal stress indicator.  

2. **Efficiency metrics**  
   - Charge efficiency ($\eta_Q$) and energy efficiency ($\eta_E$) implemented via simple ratio formulas.  
   - Early detection of parasitic losses through variance in $\eta_Q$ and decline in $\eta_E$.  

3. **Internal resistance estimation**  
   - Linear regression between voltage and current during transient steps.  
   - Median resistance per cycle, constrained between 0–1 Ω.  

4. **Knee point detection**  
   - Piecewise linear regression with penalty-based cost function.  
   - Dual-criterion detection using $\eta_E$ trends and rolling variance of $\eta_Q$.  

5. **Cross-domain validation**  
   - Integration of electrochemical, thermal, and resistance signatures.  
   - Reduction of false positives in prognostic detection.  

---

## Datasets
The repository provides three annotated datasets derived from lithium-ion cycling experiments.  

- **Cell1**  
  - Initial capacity: 8.516 Ah  
  - Final capacity: 1.843 Ah  
  - Capacity retention: 21.6% after 250 cycles  
  - Knee point: cycle 26  

- **Cell2**  
  - Initial capacity: 3.125 Ah  
  - Final capacity: 2.045 Ah  
  - Capacity retention: 65.5% after 250 cycles  
  - Knee point: cycle 47  

- **Cell3**  
  - Initial capacity: 3.021 Ah  
  - Final capacity: 1.819 Ah  
  - Capacity retention: 60.2% after 250 cycles  
  - Knee point: cycle 47  

Each dataset includes:  
- Time ($t$)  
- Cell voltage ($E_{\text{cell}}$)  
- Current ($I$)  
- Charge/discharge capacity ($Q_{\text{chg}}, Q_{\text{dis}}$)  
- Charge/discharge energy ($E_{\text{chg}}, E_{\text{dis}}$)  
- Temperature ($T$)  
- Cycle index ($k$)  

---

## Installation
Clone this repository:
```bash
git clone https://github.com/YourUsername/LiIon_Battery_SoH_RUL_Diagnostics.git
cd LiIon_Battery_SoH_RUL_Diagnostics



Install dependencies:

pip install -r requirements.txt


Recommended environment:

Python 3.9+

Jupyter Notebook or Google Colab for visualization

Usage
1. Preprocessing
python scripts/preprocessing.py --input data/Cell1_data.xlsx --output results/Cell1_clean.csv

2. Efficiency Metrics
python scripts/efficiency_metrics.py --data results/Cell1_clean.csv --save results/Cell1_efficiency.csv

3. Resistance Estimation
python scripts/resistance_estimation.py --data results/Cell1_clean.csv --save results/Cell1_resistance.csv

4. Knee Point Detection
python scripts/knee_detection.py --efficiency results/Cell1_efficiency.csv --output results/Cell1_knee.json

5. Visualization
python scripts/visualization_tools.py --data results/Cell1_efficiency.csv

Results

The analysis identified distinct degradation behaviors:

Cell1: Rapid deterioration with early knee detection at cycle 26; strong thermal coupling and efficiency decline dominated the degradation.

Cell2: Slower degradation, maintaining 65.5% retention after 250 cycles; knee detected at cycle 47; stable thermal behavior with minimal high-temperature exposure.

Cell3: Similar to Cell2, retaining 60.2% after 250 cycles; knee point at cycle 47; degradation largely electrochemical rather than thermal.

The framework demonstrated cross-domain consistency, confirming the reliability of combined diagnostics in detecting degradation inflections.

Figures

Representative figures are included in results/figures/.

Capacity degradation profiles (Cell1, Cell2, Cell3)

Charge and energy efficiency trajectories

Internal resistance evolution plots

Thermal statistics including $T_{\text{mean}}, T_{95}$, and $f_{T>40}$

Knee point detection plots with marked inflection cycles

Applications

The framework and datasets are relevant for:

Battery Management Systems (BMS): real-time health monitoring and predictive maintenance.

Electric Vehicles (EVs): range forecasting and safety assurance.

Grid Storage: life extension and second-life utilization.

Aerospace Systems: mission-critical safety and reliability.

Academic Research: benchmarking machine learning and hybrid prognostic models.

Citation

If you use this repository, please cite the associated research article:

Author(s). Title of the Paper. Journal Name, Year, Volume(Issue), Pages. 
DOI: [to be inserted once available]


BibTeX:

@article{YourCitationKey,
  author    = {Bairwa, Bansilal and Co-Authors},
  title     = {Integrated Efficiency, Thermal, and Resistance Diagnostics for Early Knee Point Detection in Lithium-Ion Cells},
  journal   = {Journal of Energy Storage},
  year      = {2025},
  volume    = {XX},
  pages     = {XXX--XXX},
  doi       = {XX.XXXX/j.jes.2025.XXXXX}
}

License

This repository is licensed under the MIT License. See the LICENSE file for details
