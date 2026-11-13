"""Stage 2 - Data validation.

Rule-based checks on the raw (string) tables:
  * missing values           * incorrect data types
  * invalid ratings          * invalid timestamps
  * duplicates               * referential integrity (interactions -> users/items)
Valid rows go to data/processed/validated_*.parquet; every invalid row goes to
data/errors/<table>_errors.csv with the reason(s) it failed. A Markdown + JSON
validation report is written to reports/.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.utils import GENRES, get_logger, get_path, load_config, save_table, write_json


def _add_error(errors: pd.Series, mask: pd.Series, reason: str) -> pd.Series:
    """Append `reason` to the error string of every row where mask is True."""
    mask = mask.fillna(False).astype(bool)
    return errors.where(~mask, errors + reason + ";")


def _is_int(s: pd.Series) -> pd.Series:
    """True where the string is a whole number (e.g. '42'), False otherwise."""
    return s.astype(str).str.fullmatch(r"-?\d+")


def validate_users(users: pd.DataFrame, vcfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check user records: ids, age range and type, gender, occupation, duplicates."""
    err = pd.Series("", index=users.index)
    err = _add_error(err, users["user_id"].str.strip() == "", "missing_user_id")
    err = _add_error(err, ~_is_int(users["user_id"]), "user_id_not_integer")
    err = _add_error(err, ~_is_int(users["age"]), "age_not_integer")
    age = pd.to_numeric(users["age"], errors="coerce")
    err = _add_error(err, (age < vcfg["user_age_min"]) | (age > vcfg["user_age_max"]), "age_out_of_range")
    err = _add_error(err, ~users["gender"].str.upper().isin(vcfg["allowed_genders"]), "invalid_gender")
    err = _add_error(err, users["occupation"].str.strip() == "", "missing_occupation")
    err = _add_error(err, users.duplicated("user_id", keep="first"), "duplicate_user_id")
    return _split(users, err)


def validate_items(items: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check item records: ids, titles, release-date format, genre flags, duplicates."""
    err = pd.Series("", index=items.index)
    err = _add_error(err, ~_is_int(items["item_id"]), "item_id_not_integer")
    err = _add_error(err, items["title"].str.strip() == "", "missing_title")
    has_date = items["release_date"].str.strip() != ""          # a missing date is allowed (imputed later)
    bad_date = pd.to_datetime(items["release_date"], format="%d-%b-%Y", errors="coerce").isna()
    err = _add_error(err, has_date & bad_date, "invalid_release_date")
    bad_genre = ~items[GENRES].isin(["0", "1"]).all(axis=1)
    err = _add_error(err, bad_genre, "invalid_genre_flag")
    err = _add_error(err, items.duplicated("item_id", keep="first"), "duplicate_item_id")
    return _split(items, err)


def validate_interactions(inter: pd.DataFrame, valid_users: pd.DataFrame,
                          valid_items: pd.DataFrame, vcfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check interactions: missing values, types, rating range, timestamps,
    exact duplicates and referential integrity against the VALID users/items."""
    err = pd.Series("", index=inter.index)
    for col in ["user_id", "item_id", "rating", "timestamp"]:
        missing = inter[col].str.strip() == ""
        err = _add_error(err, missing, f"missing_{col}")
        err = _add_error(err, ~missing & ~_is_int(inter[col]), f"{col}_not_integer")

    rating = pd.to_numeric(inter["rating"], errors="coerce")
    err = _add_error(err, rating.notna() & ~rating.between(vcfg["rating_min"], vcfg["rating_max"]),
                     "rating_out_of_range")

    ts = pd.to_numeric(inter["timestamp"], errors="coerce")
    min_ts = pd.Timestamp(vcfg["min_timestamp"], tz="UTC").timestamp()
    now_ts = datetime.now(timezone.utc).timestamp()
    err = _add_error(err, ts.notna() & (ts < min_ts), "timestamp_before_min")
    err = _add_error(err, ts.notna() & (ts > now_ts), "timestamp_in_future")

    err = _add_error(err, inter.duplicated(keep="first"), "exact_duplicate")

    has_user = inter["user_id"].str.strip() != ""
    has_item = inter["item_id"].str.strip() != ""
    err = _add_error(err, has_user & ~inter["user_id"].isin(valid_users["user_id"]), "unknown_user_id")
    err = _add_error(err, has_item & ~inter["item_id"].isin(valid_items["item_id"]), "unknown_item_id")
    return _split(inter, err)


def _split(df: pd.DataFrame, err: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a table into valid rows and invalid rows (with an `error_reason` column)."""
    bad = err != ""
    invalid = df[bad].copy()
    invalid["error_reason"] = err[bad].str.rstrip(";")
    return df[~bad].copy(), invalid


def _cast_valid(users: pd.DataFrame, items: pd.DataFrame, inter: pd.DataFrame):
    """Convert validated string columns to their proper numeric types."""
    users = users.astype({"user_id": "int64", "age": "int64"})
    items = items.astype({"item_id": "int64", **{g: "int8" for g in GENRES}})
    inter = inter.astype({"user_id": "int64", "item_id": "int64", "rating": "int64", "timestamp": "int64"})
    return users, items, inter


def _rule_counts(invalid: pd.DataFrame) -> dict:
    """Count how many rows failed each rule (a row can fail several rules)."""
    if invalid.empty:
        return {}
    return invalid["error_reason"].str.split(";").explode().value_counts().to_dict()


def write_report(results: dict, path_md, path_json) -> None:
    """Write the validation report in Markdown (for humans) and JSON (for machines)."""
    lines = ["# Data Validation Report", "",
             f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}", "",
             "| Table | Rows checked | Valid | Invalid | Invalid % |", "|---|---|---|---|---|"]
    for t, r in results.items():
        lines.append(f"| {t} | {r['total']} | {r['valid']} | {r['invalid']} | {r['invalid_pct']:.3f}% |")
    for t, r in results.items():
        lines += ["", f"## {t}: failed rules", "", "| Rule | Rows |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in r["rules"].items()] or ["| (none) | 0 |"]
    lines += ["", "Invalid rows are stored with an `error_reason` column in `data/errors/`."]
    path_md.write_text("\n".join(lines), encoding="utf-8")
    write_json(results, path_json)


def run_validation(cfg: dict | None = None) -> dict:
    """Run every validation rule, save valid/invalid splits and the report."""
    cfg = cfg or load_config()
    logger = get_logger("validation", cfg)
    raw, processed, errors_dir, reports = (get_path(cfg, k) for k in ["raw", "processed", "errors", "reports"])
    vcfg = cfg["validation"]

    read = lambda n: pd.read_csv(raw / f"{n}.csv", dtype=str, keep_default_na=False)
    users, items, inter = read("users"), read("items"), read("interactions")

    v_users, bad_users = validate_users(users, vcfg)
    v_items, bad_items = validate_items(items)
    v_inter, bad_inter = validate_interactions(inter, v_users, v_items, vcfg)

    results = {}
    for name, total, valid, bad in [("users", users, v_users, bad_users),
                                    ("items", items, v_items, bad_items),
                                    ("interactions", inter, v_inter, bad_inter)]:
        bad.to_csv(errors_dir / f"{name}_errors.csv", index=False)
        results[name] = {"total": len(total), "valid": len(valid), "invalid": len(bad),
                         "invalid_pct": 100 * len(bad) / max(len(total), 1),
                         "rules": _rule_counts(bad)}
        logger.info("%-12s total=%6d valid=%6d invalid=%4d  rules=%s", name, len(total),
                    len(valid), len(bad), results[name]["rules"])

    v_users, v_items, v_inter = _cast_valid(v_users, v_items, v_inter)
    for name, df in [("validated_users", v_users), ("validated_items", v_items),
                     ("validated_interactions", v_inter)]:
        save_table(df, processed, name)

    write_report(results, reports / "validation_report.md", reports / "validation_report.json")
    logger.info("Validation report written to reports/validation_report.md")
    return {"stage": "validation", **{k: {kk: v[kk] for kk in ("valid", "invalid")} for k, v in results.items()}}


if __name__ == "__main__":
    run_validation()
