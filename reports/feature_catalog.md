# Feature Catalog

Features are computed from training data only, as of the train/test cutoff. User table: 943 rows x 32 columns; item table: 1682 rows x 30 columns.

| Feature | Entity | Type | Description | Source | Used by |
|---|---|---|---|---|---|
| `user_rating_count` | user | int | Number of ratings the user gave in the training period | train | Analysis / reports |
| `user_avg_rating` | user | float | Mean rating given by the user (global mean for new users) | train | Analysis / reports |
| `user_rating_std` | user | float | Standard deviation of the user's ratings | train | Analysis / reports |
| `user_positive_ratio` | user | float | Share of the user's ratings that are >= 4 | train | Analysis / reports |
| `user_active_days` | user | int | Days between the user's first and last training rating | train | Analysis / reports |
| `user_days_since_last` | user | int | Days from the user's last rating to the split cutoff (-1 = never) | train | Analysis / reports |
| `user_preferred_genre` | user | category | Genre the user rated most often (segment genre for new users) | train + items | Analysis / reports |
| `user_genre_affinity_<genre>` | user | float x19 | Share of the user's ratings per genre (new users: average of their gender/age segment) | train + items + users | Hybrid |
| `is_cold_start_user` | user | int | 1 if the user has no training interactions | train | Hybrid, MF (fallback) |
| `age, gender, occupation, age_group` | user | mixed | Cleaned demographic attributes | users | Analysis / reports |
| `item_rating_count` | item | int | Number of training ratings of the item | train | Analysis / reports |
| `item_avg_rating` | item | float | Mean training rating of the item (global mean if unrated) | train | Analysis / reports |
| `item_bayesian_avg` | item | float | Rating mean shrunk towards the global mean (damping = 10) | train | Popularity |
| `item_popularity_score` | item | float | log(1 + rating count) scaled to [0, 1] | train | Hybrid, Popularity |
| `item_popularity_rank` | item | int | Rank by rating count (1 = most popular) | train | Analysis / reports |
| `item_days_since_last` | item | int | Days from the item's last rating to the cutoff (-1 = never) | train | Analysis / reports |
| `item_release_year` | item | int | Release year (missing years imputed with the median) | items | Analysis / reports |
| `item_age_years` | item | int | Cutoff year minus release year | items | Analysis / reports |
| `<19 genre flags>, n_genres` | item | int | One-hot genre indicators and number of genres | items | Hybrid, Evaluation (diversity) |
| `user_item_genre_match` | interaction | float | Mean user genre affinity over the item's genres | user + item features | Hybrid |
| `user_item_rating_deviation` | interaction | float | Rating minus the user's mean rating (training pairs) | train | Analysis / reports |
| `item_avg_minus_user_avg` | interaction | float | Item mean rating minus user mean rating | user + item features | Analysis / reports |