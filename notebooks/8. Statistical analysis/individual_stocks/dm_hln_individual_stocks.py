import pandas as pd
import numpy as np
import pickle
import os
from scipy import stats

STOCKS = ['AAPL', 'NFLX', 'MSFT', 'CRM', 'AMZN']
RESULTS_BASE = '/Users/ive/Development/Privatno/PrimoGPT/notebooks/7. Results'
OUTPUT_DIR = '/Users/ive/Development/Privatno/PrimoGPT/notebooks/8. Statistical Analysis/individual_stocks'

SECOND_BEST = {
    'AAPL': ('Momentum', 'momentum_strategy.pkl', 'Value'),
    'NFLX': ('Buy & Hold', 'finrl_buy_and_hold.pkl', 'close'),
    'MSFT': ('FinRL', 'finrl_df_account_value_ppo.pkl', 'account_value'),
    'CRM': ('Price-MA', 'p_ma_strategy.pkl', 'Value'),
    'AMZN': ('FinRL', 'finrl_df_account_value_ppo.pkl', 'account_value')
}

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def prepare_datetime_index(df):
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    elif 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df.sort_index()

def calculate_returns(series):
    return series.pct_change().dropna()

def dm_test_hln(actual, pred1, pred2):
    T = len(actual)
    e1 = actual - pred1
    e2 = actual - pred2
    d = e1**2 - e2**2
    d_mean = np.mean(d)

    def autocov(x, lag):
        if lag == 0:
            return np.var(x, ddof=1)
        if lag >= len(x):
            return 0
        return np.cov(x[:-lag], x[lag:], ddof=1)[0, 1]

    gamma_0 = autocov(d, 0)
    gamma_sum = sum(autocov(d, lag) for lag in range(1, min(1, T-1)))
    d_var = (gamma_0 + 2 * gamma_sum) / T

    if d_var <= 0:
        return 0.0, 1.0

    dm_stat = d_mean / np.sqrt(d_var)
    hln_adj = np.sqrt((T + 1 - 2 + (1-1)/T) / T)
    dm_stat = dm_stat * hln_adj
    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_stat), df=T-1))

    return dm_stat, p_value

def main():
    results = []

    for idx, stock in enumerate(STOCKS, 1):
        stock_dir = os.path.join(RESULTS_BASE, f'{idx}. {stock}', 'data')
        if not os.path.exists(stock_dir):
            continue

        # Load PrimoRL
        primorl_file = os.path.join(stock_dir, f'{stock.lower()}_primorl_df_account_value_ppo.pkl')
        if not os.path.exists(primorl_file):
            continue
        primorl_df = prepare_datetime_index(load_pickle(primorl_file))

        # Load second-best
        second_name, second_file, value_col = SECOND_BEST[stock]
        second_path = os.path.join(stock_dir, f'{stock.lower()}_{second_file}')
        if not os.path.exists(second_path):
            continue
        second_df = prepare_datetime_index(load_pickle(second_path))

        # Load stock prices
        bh_file = os.path.join(stock_dir, f'{stock.lower()}_finrl_buy_and_hold.pkl')
        if not os.path.exists(bh_file):
            continue
        stock_prices = prepare_datetime_index(load_pickle(bh_file))['close']

        # Calculate returns
        stock_ret = calculate_returns(stock_prices)
        primorl_ret = calculate_returns(primorl_df['account_value'])
        second_ret = calculate_returns(second_df[value_col])

        # Align
        common = stock_ret.index.intersection(primorl_ret.index).intersection(second_ret.index)
        actual = stock_ret.loc[common].values
        primorl = primorl_ret.loc[common].values
        second = second_ret.loc[common].values

        # Test
        dm_stat, p_val = dm_test_hln(actual, primorl, second)
        winner = 'PrimoRL' if dm_stat < 0 else second_name
        significant = 'Yes' if p_val < 0.05 else 'No'

        results.append({
            'Stock': stock,
            'Comparison': f'PrimoRL vs {second_name}',
            'DM Statistic': dm_stat,
            'p-value': p_val,
            'Significant (α=0.05)': significant,
            'Winner': winner
        })

    df = pd.DataFrame(results)

    # Save
    md_file = os.path.join(OUTPUT_DIR, 'dm_hln_individual_stocks_results.md')
    with open(md_file, 'w') as f:
        f.write("# Diebold-Mariano + HLN Test: Individual Stocks\n\n")
        f.write("## Test Results\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")

    print(f"✓ Results saved: {md_file}")

if __name__ == "__main__":
    main()
