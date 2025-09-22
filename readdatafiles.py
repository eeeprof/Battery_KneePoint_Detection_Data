import pandas as pd
import matplotlib.pyplot as plt

# -------- Read the Excel file --------
df = pd.read_excel("DataFile1.xlsx")

# -------- Global font settings --------
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 30
})

# -------- Plot 1: Voltage vs Time --------
plt.figure(figsize=(16, 8))
plt.plot(df['time_s'], df['Ecell_V'], color="blue", label="Cell Voltage (V)")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.legend()
plt.grid(True)
plt.savefig("Voltage_vs_Time.png", dpi=1000, bbox_inches="tight")
plt.show()

# -------- Plot 2: Current vs Time --------
plt.figure(figsize=(14, 10))
plt.plot(df['time_s'], df['I_mA'], color="red", label="Current (mA)")
plt.xlabel("Time (s)")
plt.ylabel("Current (mA)")
plt.legend()
plt.grid(True)
plt.savefig("Current_vs_Time.png", dpi=1000, bbox_inches="tight")
plt.show()

# -------- Plot 3: Energy Charge vs Time --------
plt.figure(figsize=(14, 10))
plt.plot(df['time_s'], df['EnergyCharge_W_h'], color="green", label="Energy Charge (Wh)")
plt.xlabel("Time (s)")
plt.ylabel("Energy (Wh)")
plt.legend()
plt.grid(True)
plt.savefig("EnergyCharge_vs_Time.png", dpi=1000, bbox_inches="tight")
plt.show()

# -------- Plot 4: Energy Discharge vs Time --------
plt.figure(figsize=(14, 10))
plt.plot(df['time_s'], df['EnergyDischarge_W_h'], color="orange", label="Energy Discharge (Wh)")
plt.xlabel("Time (s)")
plt.ylabel("Energy (Wh)")
plt.legend()
plt.grid(True)
plt.savefig("EnergyDischarge_vs_Time.png", dpi=1000, bbox_inches="tight")
plt.show()

# -------- Plot 5: Temperature vs Time --------
plt.figure(figsize=(14, 10))
plt.plot(df['time_s'], df['Temperature__C'], color="purple", label="Temperature (°C)")
plt.xlabel("Time (s)")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True)
plt.savefig("Temperature_vs_Time.png", dpi=1000, bbox_inches="tight")
plt.show()

# -------- Plot 6: Capacity vs Cycle --------
plt.figure(figsize=(14, 10))
plt.plot(df['cycleNumber'], df['QCharge_mA_h'], color="blue", label="Q Charge (mAh)")
plt.plot(df['cycleNumber'], df['QDischarge_mA_h'], color="red", label="Q Discharge (mAh)")
plt.xlabel("Cycle Number")
plt.ylabel("Capacity (mAh)")
plt.legend()
plt.grid(True)
plt.savefig("Capacity_vs_Cycle.png", dpi=1000, bbox_inches="tight")
plt.show()
