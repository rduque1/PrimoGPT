# Diebold-Mariano + HLN Test: Individual Stocks

## Test Results

| Stock   | Comparison            |   DM Statistic |     p-value | Significant (α=0.05)   | Winner     |
|:--------|:----------------------|---------------:|------------:|:-----------------------|:-----------|
| AAPL    | PrimoRL vs Momentum   |      -2.40809  | 0.0173287   | Yes                    | PrimoRL    |
| NFLX    | PrimoRL vs Buy & Hold |       1.88146  | 0.0619702   | No                     | Buy & Hold |
| MSFT    | PrimoRL vs FinRL      |       3.65598  | 0.000360761 | Yes                    | FinRL      |
| CRM     | PrimoRL vs Price-MA   |      -5.69058  | 7.05642e-08 | Yes                    | PrimoRL    |
| AMZN    | PrimoRL vs FinRL      |       0.165688 | 0.86864     | No                     | FinRL      |
