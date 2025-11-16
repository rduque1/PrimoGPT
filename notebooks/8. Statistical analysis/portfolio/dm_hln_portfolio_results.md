# Diebold-Mariano + HLN Test: Portfolio

## Test Results

| Comparison                   |   DM Statistic |     p-value | Significant (α=0.05)   | Winner        |
|:-----------------------------|---------------:|------------:|:-----------------------|:--------------|
| PrimoRL PPO vs FinRL SAC     |       -2.33104 | 0.0211689   | Yes                    | PrimoRL PPO   |
| PrimoRL PPO vs Mean-Variance |        4.79317 | 4.11676e-06 | Yes                    | Mean-Variance |

## Performance Metrics

|               |   Mean Return (%) |   Std Dev (%) |   Sharpe Ratio |   Cumulative Return (%) |   Max Drawdown (%) |
|:--------------|------------------:|--------------:|---------------:|------------------------:|-------------------:|
| PrimoRL_PPO   |          0.183826 |       1.70783 |        1.70869 |                 27.1485 |          -11.4496  |
| FinRL_SAC     |          0.177804 |       2.18308 |        1.29292 |                 24.5033 |          -16.5026  |
| Mean_Variance |          0.152133 |       1.46599 |        1.64738 |                 22.2425 |           -9.22443 |
