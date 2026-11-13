# Data Validation Report

Generated: 2026-11-13T17:24:01+00:00

| Table | Rows checked | Valid | Invalid | Invalid % |
|---|---|---|---|---|
| users | 948 | 943 | 5 | 0.527% |
| items | 1684 | 1682 | 2 | 0.119% |
| interactions | 100060 | 100000 | 60 | 0.060% |

## users: failed rules

| Rule | Rows |
|---|---|
| age_out_of_range | 2 |
| invalid_gender | 1 |
| age_not_integer | 1 |
| missing_occupation | 1 |
| duplicate_user_id | 1 |

## items: failed rules

| Rule | Rows |
|---|---|
| missing_title | 1 |
| duplicate_item_id | 1 |

## interactions: failed rules

| Rule | Rows |
|---|---|
| rating_out_of_range | 6 |
| rating_not_integer | 6 |
| missing_user_id | 6 |
| unknown_user_id | 6 |
| unknown_item_id | 6 |
| timestamp_in_future | 6 |
| timestamp_before_min | 6 |
| timestamp_not_integer | 6 |
| exact_duplicate | 6 |
| missing_rating | 6 |

Invalid rows are stored with an `error_reason` column in `data/errors/`.