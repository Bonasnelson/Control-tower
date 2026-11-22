# Reason-code hit rate, model v1 on E10 (field surprisal, top-3)

hit_any: share of injected records whose top-3 fields include a corrupted field. hit_share: mean share of the top-3 that are corrupted fields. chance_*: the same for 3 random fields. Mean ± std over seeds. For local anomalies at most 2 fields are uncorrupted, so hit_any = 1 by construction; read hit_share.

| city | kind | seeds | records per seed | hit_any | hit_share | chance_hit_any | chance_hit_share |
|---|---|---|---|---|---|---|---|
| philadelphia | global | 3 | 10 | 0.967 ± 0.058 | 0.622 ± 0.107 | 0.821 ± 0.000 | 0.375 ± 0.000 |
| philadelphia | local | 3 | 10 | 1.000 ± 0.000 | 0.878 ± 0.096 | 1.000 ± 0.000 | 0.750 ± 0.000 |
| chicago | global | 3 | 10 | 0.967 ± 0.058 | 0.544 ± 0.084 | 0.886 ± 0.000 | 0.429 ± 0.000 |
| chicago | local | 3 | 10 | 1.000 ± 0.000 | 0.856 ± 0.084 | 1.000 ± 0.000 | 0.714 ± 0.000 |
