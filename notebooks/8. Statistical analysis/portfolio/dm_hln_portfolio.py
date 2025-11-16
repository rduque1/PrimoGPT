import pandas as pd
import numpy as np
import pickle
import os
from scipy import stats

PORTFOLIO_DIR = '/Users/ive/Development/Privatno/PrimoGPT/notebooks/7. Results/6. PORTFOLIO/data'
OUTPUT_DIR = '/Users/ive/Development/Privatno/PrimoGPT/notebooks/8. Statistical Analysis/portfolio'

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

def calculate_performance_metrics(returns):
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)
    sharpe = mean_return / std_return * np.sqrt(252) if std_return != 0 else 0
    cumulative_return = np.prod(1 + returns) - 1
    cumulative_returns = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = np.min(drawdown)

    return {
        'Mean Return (%)': mean_return * 100,
        'Std Dev (%)': std_return * 100,
        'Sharpe Ratio': sharpe,
        'Cumulative Return (%)': cumulative_return * 100,
        'Max Drawdown (%)': max_drawdown * 100
    }

def main():
    # Load strategies
    primorl_ppo = prepare_datetime_index(load_pickle(os.path.join(PORTFOLIO_DIR, 'primorl_df_account_value_ppo.pkl')))
    finrl_sac = prepare_datetime_index(load_pickle(os.path.join(PORTFOLIO_DIR, 'finrl_df_account_value_sac.pkl')))
    mean_var = prepare_datetime_index(load_pickle(os.path.join(PORTFOLIO_DIR, 'primorl_mean_var.pkl')))

    # Calculate returns
    primorl_ppo_ret = calculate_returns(primorl_ppo['account_value'])
    finrl_sac_ret = calculate_returns(finrl_sac['account_value'])
    mean_var_ret = calculate_returns(mean_var['Mean Var'])

    # Align
    common = primorl_ppo_ret.index.intersection(finrl_sac_ret.index).intersection(mean_var_ret.index)
    actual = mean_var_ret.loc[common].values  # Use Mean-Variance as proxy
    primorl = primorl_ppo_ret.loc[common].values
    finrl = finrl_sac_ret.loc[common].values
    mv = mean_var_ret.loc[common].values

    # Tests
    results = []

    # Test 1: PrimoRL PPO vs FinRL SAC
    dm_stat, p_val = dm_test_hln(actual, primorl, finrl)
    winner = 'PrimoRL PPO' if dm_stat < 0 else 'FinRL SAC'
    results.append({
        'Comparison': 'PrimoRL PPO vs FinRL SAC',
        'DM Statistic': dm_stat,
        'p-value': p_val,
        'Significant (α=0.05)': 'Yes' if p_val < 0.05 else 'No',
        'Winner': winner
    })

    # Test 2: PrimoRL PPO vs Mean-Variance
    dm_stat, p_val = dm_test_hln(actual, primorl, mv)
    winner = 'PrimoRL PPO' if dm_stat < 0 else 'Mean-Variance'
    results.append({
        'Comparison': 'PrimoRL PPO vs Mean-Variance',
        'DM Statistic': dm_stat,
        'p-value': p_val,
        'Significant (α=0.05)': 'Yes' if p_val < 0.05 else 'No',
        'Winner': winner
    })

    # Performance metrics
    perf_metrics = {
        'PrimoRL_PPO': calculate_performance_metrics(primorl),
        'FinRL_SAC': calculate_performance_metrics(finrl),
        'Mean_Variance': calculate_performance_metrics(mv)
    }

    # Save
    md_file = os.path.join(OUTPUT_DIR, 'dm_hln_portfolio_results.md')
    with open(md_file, 'w') as f:
        f.write("# Diebold-Mariano + HLN Test: Portfolio\n\n")
        f.write("## Test Results\n\n")
        f.write(pd.DataFrame(results).to_markdown(index=False))
        f.write("\n\n## Performance Metrics\n\n")
        f.write(pd.DataFrame.from_dict(perf_metrics, orient='index').to_markdown())
        f.write("\n")

    print(f"✓ Results saved: {md_file}")

if __name__ == "__main__":
    main()
