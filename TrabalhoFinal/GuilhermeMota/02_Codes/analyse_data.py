import os
import pandas as pd


power_flow_data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'MT_dsa_history_data_with_labels.csv'))

volt_viol = power_flow_data[power_flow_data['VltCode'] == 1].shape[0]
volt_safe = power_flow_data[power_flow_data['VltCode'] == 0].shape[0]

thermal_viol = power_flow_data[power_flow_data['ThrCode'] == 1].shape[0]
thermal_safe = power_flow_data[power_flow_data['ThrCode'] == 0].shape[0]

print(f"Voltage safe {volt_safe} and violated {volt_viol} cases")
print(f"Thermal safe {thermal_safe} and violated {thermal_viol} cases")