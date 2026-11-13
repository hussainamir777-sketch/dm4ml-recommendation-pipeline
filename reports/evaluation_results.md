# Model Evaluation Results

Time-based split; relevant = test rating >= 4; K = 10; 290 evaluated users (98 warm, 192 cold-start).

## All test users

| model | precision_at_10 | recall_at_10 | ndcg_at_10 | hit_rate_at_10 | coverage | diversity | rmse | mae |
|---|---|---|---|---|---|---|---|---|
| Popularity (baseline) | 0.2448 | 0.0831 | 0.2706 | 0.7103 | 0.044 | 0.6955 | 1.0292 | 0.8296 |
| Matrix Factorization (SVD) | 0.2424 | 0.0815 | 0.2706 | 0.7069 | 0.1778 | 0.7019 | 1.0235 | 0.8226 |
| Hybrid (MF + features) | 0.2431 | 0.0813 | 0.2707 | 0.7172 | 0.1623 | 0.6389 | 1.0235 | 0.8226 |

## By user segment

| model | segment | users | precision_at_10 | recall_at_10 | ndcg_at_10 | hit_rate_at_10 | coverage | diversity |
|---|---|---|---|---|---|---|---|---|
| Popularity (baseline) | warm | 98 | 0.1235 | 0.1 | 0.1533 | 0.4796 | 0.044 | 0.7052 |
| Popularity (baseline) | cold-start | 192 | 0.3068 | 0.0745 | 0.3305 | 0.8281 | 0.0059 | 0.6905 |
| Matrix Factorization (SVD) | warm | 98 | 0.1163 | 0.0952 | 0.1533 | 0.4694 | 0.1778 | 0.7243 |
| Matrix Factorization (SVD) | cold-start | 192 | 0.3068 | 0.0745 | 0.3305 | 0.8281 | 0.0059 | 0.6905 |
| Hybrid (MF + features) | warm | 98 | 0.1224 | 0.0957 | 0.158 | 0.4898 | 0.1623 | 0.5572 |
| Hybrid (MF + features) | cold-start | 192 | 0.3047 | 0.074 | 0.3282 | 0.8333 | 0.0077 | 0.6807 |