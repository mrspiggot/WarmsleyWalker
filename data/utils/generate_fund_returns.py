
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.formatting.rule import ColorScaleRule

# Create date range
dates = pd.date_range(start='2003-01-01', end='2023-12-31', freq='ME')
data = pd.DataFrame(index=dates)

# Initialize starting values
data['MSCI_SMID'] = 100.0
data['HFRI'] = 100.0
data['Fund'] = 100.0

# Define crisis periods
crisis_periods = {
    '2008-09': ('2008-09-01', '2008-12-31', -32, -12, -8),  # GFC
    '2011-08': ('2011-08-01', '2011-09-30', -15, -6, -4),  # Sovereign Debt
    '2016-06': ('2016-06-01', '2016-07-31', -8, -3, -2),  # Brexit
    '2020-03': ('2020-03-01', '2020-03-31', -20, -6, -4),  # COVID
    '2022-01': ('2022-01-01', '2022-06-30', -18, -8, -6)  # Tech Selloff
}

# Generate monthly returns
for i in range(len(dates) - 1):
    current_date = dates[i]

    # Base case: normal market conditions
    msci_return = np.random.normal(0.008, 0.03)  # 8% annual return, 10% vol
    hfri_return = np.random.normal(0.006, 0.02)  # 6% annual return, 7% vol
    fund_return = np.random.normal(0.009, 0.025)  # 9% annual return, 8.5% vol

    # Crisis periods modifications
    for crisis_start, (start_date, end_date, msci_drop, hfri_drop, fund_drop) in crisis_periods.items():
        if pd.Timestamp(start_date) <= current_date <= pd.Timestamp(end_date):
            crisis_severity = np.random.uniform(0.8, 1.2)
            msci_return = msci_drop / 100 * crisis_severity
            hfri_return = hfri_drop / 100 * crisis_severity
            fund_return = fund_drop / 100 * crisis_severity

    # Update values
    data.loc[dates[i + 1], 'MSCI_SMID'] = data.loc[dates[i], 'MSCI_SMID'] * (1 + msci_return)
    data.loc[dates[i + 1], 'HFRI'] = data.loc[dates[i], 'HFRI'] * (1 + hfri_return)
    data.loc[dates[i + 1], 'Fund'] = data.loc[dates[i], 'Fund'] * (1 + fund_return)

# Calculate returns
data['MSCI_SMID_Return'] = data['MSCI_SMID'].pct_change()
data['HFRI_Return'] = data['HFRI'].pct_change()
data['Fund_Return'] = data['Fund'].pct_change()

# Calculate rolling Sharpe ratios (12-month window)
risk_free_rate = 0.02 / 12  # Assuming 2% annual risk-free rate
for col in ['MSCI_SMID', 'HFRI', 'Fund']:
    returns = data[f'{col}_Return']
    rolling_excess_returns = returns - risk_free_rate
    rolling_std = returns.rolling(12).std() * np.sqrt(12)
    data[f'{col}_Sharpe'] = rolling_excess_returns.rolling(12).mean() * np.sqrt(12) / rolling_std

# Create Excel writer object
excel_file = 'Fund_Performance_Data.xlsx'
with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
    # Sheet 1: Monthly Performance Data
    monthly_data = pd.DataFrame({
        'MSCI_SMID': data['MSCI_SMID'],
        'MSCI_Return': data['MSCI_SMID_Return'],
        'MSCI_Sharpe': data['MSCI_SMID_Sharpe'],
        'HFRI': data['HFRI'],
        'HFRI_Return': data['HFRI_Return'],
        'HFRI_Sharpe': data['HFRI_Sharpe'],
        'Fund': data['Fund'],
        'Fund_Return': data['Fund_Return'],
        'Fund_Sharpe': data['Fund_Sharpe']
    }).round(4)

    monthly_data.to_excel(writer, sheet_name='Monthly Data', index=True)

    # Sheet 2: Annual Performance
    annual_returns = pd.DataFrame({
        'MSCI_Return': data['MSCI_SMID_Return'].resample('Y').apply(
            lambda x: (1 + x).prod() - 1),
        'HFRI_Return': data['HFRI_Return'].resample('Y').apply(
            lambda x: (1 + x).prod() - 1),
        'Fund_Return': data['Fund_Return'].resample('Y').apply(
            lambda x: (1 + x).prod() - 1)
    }).round(4) * 100  # Convert to percentages

    annual_returns.to_excel(writer, sheet_name='Annual Returns', index=True)

    # Sheet 3: Rolling Analytics
    rolling_metrics = pd.DataFrame({
        'MSCI_3Y_Return': data['MSCI_SMID_Return'].rolling(36).apply(
            lambda x: (1 + x).prod() - 1),
        'HFRI_3Y_Return': data['HFRI_Return'].rolling(36).apply(
            lambda x: (1 + x).prod() - 1),
        'Fund_3Y_Return': data['Fund_Return'].rolling(36).apply(
            lambda x: (1 + x).prod() - 1),
        'MSCI_3Y_Vol': data['MSCI_SMID_Return'].rolling(36).std() * np.sqrt(12),
        'HFRI_3Y_Vol': data['HFRI_Return'].rolling(36).std() * np.sqrt(12),
        'Fund_3Y_Vol': data['Fund_Return'].rolling(36).std() * np.sqrt(12)
    }).round(4)

    rolling_metrics.to_excel(writer, sheet_name='Rolling Analytics', index=True)

    # Sheet 4: Summary Statistics
    summary_stats = pd.DataFrame({
        'Metric': ['Total Return', 'Annualized Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown'],
        'MSCI_SMID': [
            f"{((data['MSCI_SMID'].iloc[-1] / data['MSCI_SMID'].iloc[0] - 1) * 100):.2f}%",
            f"{((pow(data['MSCI_SMID'].iloc[-1] / data['MSCI_SMID'].iloc[0], 1 / 20) - 1) * 100):.2f}%",
            f"{(data['MSCI_SMID_Return'].std() * np.sqrt(12) * 100):.2f}%",
            f"{(data['MSCI_SMID_Return'].mean() / data['MSCI_SMID_Return'].std() * np.sqrt(12)):.2f}",
            f"{((data['MSCI_SMID'] / data['MSCI_SMID'].expanding().max() - 1).min() * 100):.2f}%"
        ],
        'HFRI': [
            f"{((data['HFRI'].iloc[-1] / data['HFRI'].iloc[0] - 1) * 100):.2f}%",
            f"{((pow(data['HFRI'].iloc[-1] / data['HFRI'].iloc[0], 1 / 20) - 1) * 100):.2f}%",
            f"{(data['HFRI_Return'].std() * np.sqrt(12) * 100):.2f}%",
            f"{(data['HFRI_Return'].mean() / data['HFRI_Return'].std() * np.sqrt(12)):.2f}",
            f"{((data['HFRI'] / data['HFRI'].expanding().max() - 1).min() * 100):.2f}%"
        ],
        'Fund': [
            f"{((data['Fund'].iloc[-1] / data['Fund'].iloc[0] - 1) * 100):.2f}%",
            f"{((pow(data['Fund'].iloc[-1] / data['Fund'].iloc[0], 1 / 20) - 1) * 100):.2f}%",
            f"{(data['Fund_Return'].std() * np.sqrt(12) * 100):.2f}%",
            f"{(data['Fund_Return'].mean() / data['Fund_Return'].std() * np.sqrt(12)):.2f}",
            f"{((data['Fund'] / data['Fund'].expanding().max() - 1).min() * 100):.2f}%"
        ]
    })

    summary_stats.to_excel(writer, sheet_name='Summary Statistics', index=False)

print(f"Data has been saved to {excel_file}")
