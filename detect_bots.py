#!/usr/bin/env python3
"""Bot or Not challenge detector.

Adaptive mode (default): dataset-normalized, confidence-tiered detector optimized for
+4/-1/-2 scoring and cross-dataset robustness.
Legacy mode: preserves older fixed-threshold behavior for compatibility checks.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


KEYWORDS = ("bot", "digest", "automatic")
URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]+")
SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class LegacyParams:
    z_thr: float
    tweet_thr: int
    dup_thr: int
    hashtag_thr: float
    cv_thr: float
    avg_len_thr: float
    score_thr: int


@dataclass(frozen=True)
class AdaptiveParams:
    tweet_rank_thr: float
    z_rank_thr: float
    dup_rank_thr: float
    dupfrac_rank_thr: float
    hashtag_rank_thr: float
    mention_rank_thr: float
    temporal_rank_thr: float
    avg_len_rank_thr: float
    short_inv_rank_thr: float
    near_dup_frac_thr: float
    high_score_thr: int
    medium_score_thr: int


@dataclass(frozen=True)
class QuantStats:
    values_sorted: Tuple[float, ...]
    p50: float
    p75: float
    p90: float
    p95: float
    iqr: float


@dataclass
class EvalResult:
    score: int
    tp: int
    fn: int
    fp: int
    precision: float
    recall: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect likely bot accounts from posts/users JSON.")
    parser.add_argument("--input-json", required=True, help="Path to dataset.posts&users.json")
    parser.add_argument(
        "--output-txt",
        default="MyTeam.detections.en.txt",
        help="Output file. One predicted bot user ID per line.",
    )
    parser.add_argument(
        "--bots-txt",
        default=None,
        help="Optional ground-truth bots file for evaluation (one user ID per line).",
    )
    parser.add_argument(
        "--tune-dataset",
        action="append",
        nargs=2,
        metavar=("DATASET_JSON", "BOTS_TXT"),
        help="Optional labeled dataset pair used for threshold tuning. Repeatable.",
    )
    parser.add_argument("--no-tune", action="store_true", help="Disable tuning and use built-in defaults.")

    # New controls
    parser.add_argument("--mode", choices=["adaptive", "legacy"], default="adaptive")
    parser.add_argument("--min-posts-for-timing", type=int, default=8)
    parser.add_argument("--min-posts-for-repetition", type=int, default=6)
    parser.add_argument("--explain-top", type=int, default=0)
    parser.add_argument("--diagnostics-csv", default=None)
    parser.add_argument("--calibration-objective", choices=["robust", "mean"], default="robust")
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    parser.add_argument("--bootstrap-frac", type=float, default=0.8)
    parser.add_argument("--bootstrap-seed", type=int, default=42)

    return parser.parse_args()


def load_dataset(json_path: Path) -> Tuple[List[dict], List[dict]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    users = data.get("users")
    posts = data.get("posts")
    if not isinstance(users, list) or not isinstance(posts, list):
        raise ValueError(f"Invalid schema in {json_path}: expected top-level 'users' and 'posts' lists")
    return users, posts


def load_bot_ids(path: Path) -> Set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def normalize_for_near_dup(text: str) -> str:
    t = (text or "").lower()
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = HASHTAG_RE.sub(" ", t)
    t = NON_ALNUM_RE.sub(" ", t)
    t = SPACE_RE.sub(" ", t).strip()
    return t


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_datetime(value: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError, AttributeError):
        return None


def quantile(values_sorted: Sequence[float], p: float) -> float:
    if not values_sorted:
        return 0.0
    if len(values_sorted) == 1:
        return values_sorted[0]
    k = (len(values_sorted) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(values_sorted) - 1)
    frac = k - lo
    return values_sorted[lo] * (1.0 - frac) + values_sorted[hi] * frac


def quantile_rank(values_sorted: Sequence[float], value: float) -> float:
    if not values_sorted:
        return 0.0
    pos = bisect.bisect_right(values_sorted, value)
    return pos / len(values_sorted)


def robust_z(value: float, stats: QuantStats, eps: float = 1e-6) -> float:
    return (value - stats.p50) / max(stats.iqr, eps)


def build_user_features(users: Sequence[dict], posts: Sequence[dict]) -> Dict[str, dict]:
    per_user_counts: Dict[str, Dict[str, int]] = {}
    per_user_near_counts: Dict[str, Dict[str, int]] = {}
    per_user_times: Dict[str, List[datetime]] = {}
    per_user_hashtag_posts: Dict[str, int] = {}
    per_user_mention_posts: Dict[str, int] = {}
    per_user_short_posts: Dict[str, int] = {}
    per_user_lengths_sum: Dict[str, int] = {}
    per_user_non_empty_posts: Dict[str, int] = {}

    for post in posts:
        uid = str(post.get("author_id", "")).strip()
        if not uid:
            continue

        text = normalize_text(str(post.get("text") or ""))
        near_text = normalize_for_near_dup(text)

        exact_bucket = per_user_counts.setdefault(uid, {})
        exact_bucket[text] = exact_bucket.get(text, 0) + 1

        near_bucket = per_user_near_counts.setdefault(uid, {})
        near_bucket[near_text] = near_bucket.get(near_text, 0) + 1

        if "#" in text:
            per_user_hashtag_posts[uid] = per_user_hashtag_posts.get(uid, 0) + 1
        if "@" in text:
            per_user_mention_posts[uid] = per_user_mention_posts.get(uid, 0) + 1

        if text:
            per_user_non_empty_posts[uid] = per_user_non_empty_posts.get(uid, 0) + 1
            text_len = len(text)
            per_user_lengths_sum[uid] = per_user_lengths_sum.get(uid, 0) + text_len
            if text_len < 40:
                per_user_short_posts[uid] = per_user_short_posts.get(uid, 0) + 1

        dt = parse_datetime(str(post.get("created_at", "")))
        if dt is not None:
            per_user_times.setdefault(uid, []).append(dt)

    features: Dict[str, dict] = {}
    for user in users:
        uid = str(user.get("id", "")).strip()
        if not uid:
            continue

        text_counts = per_user_counts.get(uid, {})
        near_counts = per_user_near_counts.get(uid, {})

        total_posts = sum(text_counts.values())
        unique_posts = len(text_counts)
        max_duplicate = max(text_counts.values()) if text_counts else 0
        duplicate_fraction = (1.0 - (unique_posts / total_posts)) if total_posts else 0.0

        near_unique_posts = len(near_counts)
        max_near_duplicate = max(near_counts.values()) if near_counts else 0
        near_duplicate_fraction = (1.0 - (near_unique_posts / total_posts)) if total_posts else 0.0
        near_collision_concentration = (max_near_duplicate / total_posts) if total_posts else 0.0

        hashtag_rate = per_user_hashtag_posts.get(uid, 0) / total_posts if total_posts else 0.0
        mention_rate = per_user_mention_posts.get(uid, 0) / total_posts if total_posts else 0.0

        non_empty_posts = per_user_non_empty_posts.get(uid, 0)
        avg_text_length = per_user_lengths_sum.get(uid, 0) / non_empty_posts if non_empty_posts else 0.0
        short_post_rate = per_user_short_posts.get(uid, 0) / non_empty_posts if non_empty_posts else 0.0

        post_times = sorted(per_user_times.get(uid, []))
        cv_gap = 99.0
        burstiness = 0.0
        if len(post_times) >= 3:
            gaps = [(post_times[i] - post_times[i - 1]).total_seconds() for i in range(1, len(post_times))]
            if gaps:
                mean_gap = sum(gaps) / len(gaps)
                if mean_gap > 0:
                    variance = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
                    cv_gap = (variance ** 0.5) / mean_gap
                burstiness = sum(1 for g in gaps if g <= 300) / len(gaps)  # <= 5 minutes

        username = str(user.get("username") or "").lower()
        description = str(user.get("description") or "").lower()
        location = str(user.get("location") or "")

        has_keyword = any(k in username or k in description for k in KEYWORDS)
        empty_description = not description.strip()
        empty_location = not location.strip()

        features[uid] = {
            "z_score": safe_float(user.get("z_score"), 0.0),
            "tweet_count": safe_int(user.get("tweet_count"), 0),
            "total_posts": total_posts,
            "max_duplicate": max_duplicate,
            "duplicate_fraction": duplicate_fraction,
            "max_near_duplicate": max_near_duplicate,
            "near_duplicate_fraction": near_duplicate_fraction,
            "near_collision_concentration": near_collision_concentration,
            "hashtag_rate": hashtag_rate,
            "mention_rate": mention_rate,
            "avg_text_length": avg_text_length,
            "short_post_rate": short_post_rate,
            "cv_gap": cv_gap,
            "burstiness": burstiness,
            "has_keyword": has_keyword,
            "empty_description": empty_description,
            "empty_location": empty_location,
            "empty_profile": empty_description or empty_location,
        }

    return features


def compute_dataset_stats(features: Dict[str, dict]) -> Dict[str, QuantStats]:
    numeric_keys = [
        "z_score",
        "tweet_count",
        "total_posts",
        "max_duplicate",
        "duplicate_fraction",
        "max_near_duplicate",
        "near_duplicate_fraction",
        "near_collision_concentration",
        "hashtag_rate",
        "mention_rate",
        "avg_text_length",
        "short_post_rate",
        "cv_gap",
        "burstiness",
    ]
    stats: Dict[str, QuantStats] = {}
    for key in numeric_keys:
        vals = sorted(float(f.get(key, 0.0)) for f in features.values())
        if not vals:
            vals = [0.0]
        p25 = quantile(vals, 0.25)
        p75 = quantile(vals, 0.75)
        stats[key] = QuantStats(
            values_sorted=tuple(vals),
            p50=quantile(vals, 0.50),
            p75=quantile(vals, 0.75),
            p90=quantile(vals, 0.90),
            p95=quantile(vals, 0.95),
            iqr=max(p75 - p25, 1e-6),
        )
    return stats


def add_normalized_views(features: Dict[str, dict], stats: Dict[str, QuantStats]) -> Dict[str, dict]:
    normalized: Dict[str, dict] = {}
    for uid, feat in features.items():
        row = dict(feat)
        for key, st in stats.items():
            val = float(feat.get(key, 0.0))
            row[f"{key}_rank"] = quantile_rank(st.values_sorted, val)
            row[f"{key}_rz"] = robust_z(val, st)
        row["cv_gap_inv_rank"] = 1.0 - row["cv_gap_rank"]
        row["short_post_rate_inv_rank"] = 1.0 - row["short_post_rate_rank"]
        normalized[uid] = row
    return normalized


def score_user_adaptive(feat: dict, params: AdaptiveParams, min_posts_for_timing: int, min_posts_for_repetition: int) -> Tuple[bool, str, List[str], int]:
    reasons: List[str] = []

    # Core group scores
    activity_points = 0
    if feat["tweet_count_rank"] >= params.tweet_rank_thr:
        activity_points += 2
        reasons.append("ACTIVITY_HIGH")
    if feat["z_score_rank"] >= params.z_rank_thr:
        activity_points += 1
        reasons.append("ZSCORE_HIGH")

    repetition_points = 0
    has_rep_support = feat["total_posts"] >= min_posts_for_repetition
    if has_rep_support:
        if feat["max_duplicate_rank"] >= params.dup_rank_thr:
            repetition_points += 2
            reasons.append("REPETITION_EXACT")
        if feat["duplicate_fraction_rank"] >= params.dupfrac_rank_thr:
            repetition_points += 1
            reasons.append("REPETITION_RATIO")
        if feat["near_duplicate_fraction"] >= params.near_dup_frac_thr:
            repetition_points += 1
            reasons.append("REPETITION_NEAR")

    temporal_points = 0
    has_time_support = feat["total_posts"] >= min_posts_for_timing
    if has_time_support:
        if feat["cv_gap_inv_rank"] >= params.temporal_rank_thr:
            temporal_points += 1
            reasons.append("TIMING_REGULAR")
        if feat["burstiness_rank"] >= 0.90:
            temporal_points += 1
            reasons.append("BURSTY_POSTING")

    style_points = 0
    if feat["hashtag_rate_rank"] >= params.hashtag_rank_thr:
        style_points += 1
        reasons.append("HASHTAG_HEAVY")
    if feat["mention_rate_rank"] >= params.mention_rank_thr:
        style_points += 1
        reasons.append("MENTION_HEAVY")
    if feat["avg_text_length_rank"] >= params.avg_len_rank_thr:
        style_points += 1
        reasons.append("LONG_POSTS")
    if feat["short_post_rate_inv_rank"] >= params.short_inv_rank_thr:
        style_points += 1
        reasons.append("LOW_SHORT_POST_RATE")

    profile_points = 0
    if feat["empty_profile"]:
        profile_points += 1
        reasons.append("PROFILE_WEAK")
    if feat["has_keyword"]:
        profile_points += 1
        reasons.append("BOT_KEYWORD")

    has_activity_core = activity_points >= 2
    has_repetition_core = repetition_points >= 2
    has_temporal_core = temporal_points >= 1
    has_core = has_activity_core or has_repetition_core or has_temporal_core

    strong_groups = int(activity_points > 0) + int(repetition_points > 0) + int(temporal_points > 0)
    core_points = activity_points + repetition_points + temporal_points + style_points
    total_points = core_points + min(profile_points, 1)  # profile is weak-only support

    # Minimum-support guard
    low_support = feat["total_posts"] < min(min_posts_for_repetition, min_posts_for_timing)
    exceptional = feat["tweet_count_rank"] >= 0.95 and feat["z_score_rank"] >= 0.80
    if low_support and not exceptional:
        return False, "insufficient_evidence", ["INSUFFICIENT_SUPPORT"], total_points

    high_conf = has_core and strong_groups >= 2 and total_points >= params.high_score_thr
    medium_conf_shape = (
        (has_activity_core and (has_repetition_core or has_temporal_core))
        or (has_repetition_core and has_temporal_core)
        or (has_activity_core and style_points >= 2)
    )
    medium_conf = has_core and medium_conf_shape and (strong_groups >= 2 or style_points >= 2) and total_points >= params.medium_score_thr

    if high_conf:
        return True, "high", reasons, total_points
    if medium_conf:
        return True, "medium", reasons, total_points
    return False, "none", reasons, total_points


def classify_user_legacy(feat: dict, params: LegacyParams) -> bool:
    high_z = feat["z_score"] >= params.z_thr
    high_tweets = feat["tweet_count"] >= params.tweet_thr
    repetitive = feat["max_duplicate"] >= params.dup_thr
    high_hashtag = feat["hashtag_rate"] >= params.hashtag_thr
    regular_timing = feat["cv_gap"] <= params.cv_thr
    long_posts = feat["avg_text_length"] >= params.avg_len_thr

    score = 0
    score += 2 if high_tweets else 0
    score += 1 if high_z else 0
    score += 2 if repetitive else 0
    score += 1 if high_hashtag else 0
    score += 1 if feat["mention_rate"] >= 0.15 else 0
    score += 1 if regular_timing else 0
    score += 1 if long_posts else 0
    score += 1 if feat["short_post_rate"] <= 0.10 else 0
    score += 1 if feat["empty_location"] else 0
    if feat["has_keyword"]:
        score += 1
    if feat["duplicate_fraction"] >= 0.12 and feat["total_posts"] >= 12:
        score += 1

    has_core = high_tweets or repetitive or (high_hashtag and regular_timing)
    return has_core and score >= params.score_thr


def predict_bots_adaptive(
    features: Dict[str, dict],
    params: AdaptiveParams,
    min_posts_for_timing: int,
    min_posts_for_repetition: int,
) -> Tuple[Set[str], Dict[str, dict], Dict[str, QuantStats], Dict[str, dict]]:
    stats = compute_dataset_stats(features)
    normalized = add_normalized_views(features, stats)

    predictions: Set[str] = set()
    explain: Dict[str, dict] = {}
    for uid, feat in normalized.items():
        pred, confidence, reasons, total_points = score_user_adaptive(
            feat,
            params,
            min_posts_for_timing=min_posts_for_timing,
            min_posts_for_repetition=min_posts_for_repetition,
        )
        if pred:
            predictions.add(uid)
        explain[uid] = {
            "predicted": pred,
            "confidence": confidence,
            "reasons": reasons,
            "total_points": total_points,
            "tweet_count_rank": round(feat["tweet_count_rank"], 4),
            "z_score_rank": round(feat["z_score_rank"], 4),
            "max_duplicate_rank": round(feat["max_duplicate_rank"], 4),
            "duplicate_fraction_rank": round(feat["duplicate_fraction_rank"], 4),
            "cv_gap_inv_rank": round(feat["cv_gap_inv_rank"], 4),
            "burstiness_rank": round(feat["burstiness_rank"], 4),
            "near_duplicate_fraction": round(feat["near_duplicate_fraction"], 4),
            "total_posts": feat["total_posts"],
        }

    return predictions, explain, stats, normalized


def predict_bots_legacy(features: Dict[str, dict], params: LegacyParams) -> Set[str]:
    return {uid for uid, feat in features.items() if classify_user_legacy(feat, params)}


def challenge_score(predicted: Set[str], actual_bots: Set[str], all_users: Iterable[str]) -> Tuple[int, int, int, int]:
    all_ids = set(all_users)
    tp = len(predicted & actual_bots)
    fn = len((actual_bots & all_ids) - predicted)
    fp = len((predicted & all_ids) - actual_bots)
    score = (4 * tp) - fn - (2 * fp)
    return score, tp, fn, fp


def build_eval(score: int, tp: int, fn: int, fp: int) -> EvalResult:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return EvalResult(score=score, tp=tp, fn=fn, fp=fp, precision=precision, recall=recall)


def evaluate_predictions(predicted: Set[str], actual: Set[str], all_user_ids: Iterable[str]) -> EvalResult:
    score, tp, fn, fp = challenge_score(predicted, actual, all_user_ids)
    return build_eval(score, tp, fn, fp)


def print_eval(label: str, ev: EvalResult, predicted_count: int) -> None:
    print(
        f"[{label}] score={ev.score} tp={ev.tp} fn={ev.fn} fp={ev.fp} "
        f"precision={ev.precision:.3f} recall={ev.recall:.3f} predicted={predicted_count}"
    )


def tune_legacy_params(labeled_sets: Sequence[Tuple[Dict[str, dict], Set[str]]], objective: str) -> LegacyParams:
    z_grid = [1.8, 2.0, 2.2, 2.4]
    tweet_grid = [50, 55, 60, 65, 70]
    dup_grid = [2, 3]
    hashtag_grid = [0.30, 0.40, 0.50]
    cv_grid = [1.1, 1.3, 1.5]
    avg_len_grid = [95.0, 100.0, 110.0]
    score_grid = [4, 5, 6]

    best_key: Optional[Tuple[float, float, float, float]] = None
    best_params: Optional[LegacyParams] = None

    for z in z_grid:
        for t in tweet_grid:
            for d in dup_grid:
                for h in hashtag_grid:
                    for cv in cv_grid:
                        for avg_len in avg_len_grid:
                            for s in score_grid:
                                params = LegacyParams(
                                    z_thr=z,
                                    tweet_thr=t,
                                    dup_thr=d,
                                    hashtag_thr=h,
                                    cv_thr=cv,
                                    avg_len_thr=avg_len,
                                    score_thr=s,
                                )
                                fold_scores: List[int] = []
                                fold_precisions: List[float] = []
                                for feats, bots in labeled_sets:
                                    preds = predict_bots_legacy(feats, params)
                                    ev = evaluate_predictions(preds, bots, feats.keys())
                                    fold_scores.append(ev.score)
                                    fold_precisions.append(ev.precision)

                                min_score = min(fold_scores)
                                mean_score = mean(fold_scores)
                                variance = mean([(s0 - mean_score) ** 2 for s0 in fold_scores]) if len(fold_scores) > 1 else 0.0
                                mean_precision = mean(fold_precisions)

                                if objective == "robust":
                                    key = (min_score, mean_score, -variance, mean_precision)
                                else:
                                    key = (mean_score, min_score, -variance, mean_precision)
                                if best_key is None or key > best_key:
                                    best_key = key
                                    best_params = params

    return best_params if best_params is not None else LegacyParams(2.2, 60, 2, 0.30, 1.3, 95.0, 4)


def tune_adaptive_params(
    labeled_sets: Sequence[Tuple[Dict[str, dict], Set[str]]],
    objective: str,
    min_posts_for_timing: int,
    min_posts_for_repetition: int,
) -> AdaptiveParams:
    tweet_rank_grid = [0.80, 0.90]
    z_rank_grid = [0.80, 0.90]
    dup_rank_grid = [0.85, 0.93]
    dupfrac_rank_grid = [0.85, 0.92]
    hashtag_rank_grid = [0.85]
    mention_rank_grid = [0.80]
    temporal_rank_grid = [0.85, 0.92]
    avg_len_rank_grid = [0.80]
    short_inv_rank_grid = [0.75]
    near_dup_grid = [0.15, 0.20]
    high_score_grid = [6, 7, 8]
    medium_score_grid = [5, 6, 7]

    best_key: Optional[Tuple[float, float, float, float]] = None
    best_params: Optional[AdaptiveParams] = None

    for tweet_rank in tweet_rank_grid:
        for z_rank in z_rank_grid:
            for dup_rank in dup_rank_grid:
                for dupfrac_rank in dupfrac_rank_grid:
                    for hashtag_rank in hashtag_rank_grid:
                        for mention_rank in mention_rank_grid:
                            for temporal_rank in temporal_rank_grid:
                                for avg_len_rank in avg_len_rank_grid:
                                    for short_inv_rank in short_inv_rank_grid:
                                        for near_dup in near_dup_grid:
                                            for high_score in high_score_grid:
                                                for medium_score in medium_score_grid:
                                                    if medium_score > high_score:
                                                        continue
                                                    params = AdaptiveParams(
                                                        tweet_rank_thr=tweet_rank,
                                                        z_rank_thr=z_rank,
                                                        dup_rank_thr=dup_rank,
                                                        dupfrac_rank_thr=dupfrac_rank,
                                                        hashtag_rank_thr=hashtag_rank,
                                                        mention_rank_thr=mention_rank,
                                                        temporal_rank_thr=temporal_rank,
                                                        avg_len_rank_thr=avg_len_rank,
                                                        short_inv_rank_thr=short_inv_rank,
                                                        near_dup_frac_thr=near_dup,
                                                        high_score_thr=high_score,
                                                        medium_score_thr=medium_score,
                                                    )
                                                    fold_scores: List[int] = []
                                                    fold_precisions: List[float] = []

                                                    # Leave-one-dataset-out style reporting objective: each dataset is a fold.
                                                    for feats, bots in labeled_sets:
                                                        preds, _ex, _st, _norm = predict_bots_adaptive(
                                                            feats,
                                                            params,
                                                            min_posts_for_timing=min_posts_for_timing,
                                                            min_posts_for_repetition=min_posts_for_repetition,
                                                        )
                                                        ev = evaluate_predictions(preds, bots, feats.keys())
                                                        fold_scores.append(ev.score)
                                                        fold_precisions.append(ev.precision)

                                                    min_score = min(fold_scores)
                                                    mean_score = mean(fold_scores)
                                                    variance = mean([(s0 - mean_score) ** 2 for s0 in fold_scores]) if len(fold_scores) > 1 else 0.0
                                                    mean_precision = mean(fold_precisions)

                                                    if objective == "robust":
                                                        key = (min_score, mean_score, -variance, mean_precision)
                                                    else:
                                                        key = (mean_score, min_score, -variance, mean_precision)

                                                    if best_key is None or key > best_key:
                                                        best_key = key
                                                        best_params = params

    return (
        best_params
        if best_params is not None
        else AdaptiveParams(
            tweet_rank_thr=0.90,
            z_rank_thr=0.90,
            dup_rank_thr=0.93,
            dupfrac_rank_thr=0.90,
            hashtag_rank_thr=0.85,
            mention_rank_thr=0.80,
            temporal_rank_thr=0.90,
            avg_len_rank_thr=0.80,
            short_inv_rank_thr=0.75,
            near_dup_frac_thr=0.20,
            high_score_thr=8,
            medium_score_thr=7,
        )
    )


def bootstrap_eval(
    features: Dict[str, dict],
    bots: Set[str],
    params: AdaptiveParams,
    min_posts_for_timing: int,
    min_posts_for_repetition: int,
    samples: int,
    frac: float,
    seed: int,
) -> None:
    if samples <= 0:
        return

    uids = list(features.keys())
    if not uids:
        return
    sample_n = max(1, int(len(uids) * max(0.05, min(frac, 1.0))))
    rng = random.Random(seed)
    scores: List[int] = []

    for _ in range(samples):
        sampled_ids = set(rng.sample(uids, k=min(sample_n, len(uids))))
        sampled_features = {uid: features[uid] for uid in sampled_ids}
        sampled_bots = bots & sampled_ids
        preds, _ex, _st, _norm = predict_bots_adaptive(
            sampled_features,
            params,
            min_posts_for_timing=min_posts_for_timing,
            min_posts_for_repetition=min_posts_for_repetition,
        )
        ev = evaluate_predictions(preds, sampled_bots, sampled_ids)
        scores.append(ev.score)

    scores_sorted = sorted(scores)
    p10 = quantile(scores_sorted, 0.10)
    p50 = quantile(scores_sorted, 0.50)
    p90 = quantile(scores_sorted, 0.90)
    print(
        "Bootstrap:",
        f"samples={samples}",
        f"frac={frac:.2f}",
        f"score_mean={mean(scores):.2f}",
        f"p10={p10:.2f}",
        f"p50={p50:.2f}",
        f"p90={p90:.2f}",
    )


def write_diagnostics_csv(path: Path, explain: Dict[str, dict], normalized: Dict[str, dict]) -> None:
    fieldnames = [
        "user_id",
        "predicted",
        "confidence",
        "total_points",
        "reasons",
        "total_posts",
        "tweet_count",
        "tweet_count_rank",
        "z_score",
        "z_score_rank",
        "max_duplicate",
        "max_duplicate_rank",
        "duplicate_fraction",
        "duplicate_fraction_rank",
        "near_duplicate_fraction",
        "cv_gap",
        "cv_gap_inv_rank",
        "burstiness",
        "burstiness_rank",
        "empty_profile",
        "has_keyword",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for uid, ex in explain.items():
            feat = normalized[uid]
            writer.writerow(
                {
                    "user_id": uid,
                    "predicted": int(bool(ex["predicted"])),
                    "confidence": ex["confidence"],
                    "total_points": ex["total_points"],
                    "reasons": "|".join(ex["reasons"]),
                    "total_posts": feat["total_posts"],
                    "tweet_count": feat["tweet_count"],
                    "tweet_count_rank": round(feat["tweet_count_rank"], 6),
                    "z_score": round(feat["z_score"], 6),
                    "z_score_rank": round(feat["z_score_rank"], 6),
                    "max_duplicate": feat["max_duplicate"],
                    "max_duplicate_rank": round(feat["max_duplicate_rank"], 6),
                    "duplicate_fraction": round(feat["duplicate_fraction"], 6),
                    "duplicate_fraction_rank": round(feat["duplicate_fraction_rank"], 6),
                    "near_duplicate_fraction": round(feat["near_duplicate_fraction"], 6),
                    "cv_gap": round(feat["cv_gap"], 6),
                    "cv_gap_inv_rank": round(feat["cv_gap_inv_rank"], 6),
                    "burstiness": round(feat["burstiness"], 6),
                    "burstiness_rank": round(feat["burstiness_rank"], 6),
                    "empty_profile": int(bool(feat["empty_profile"])),
                    "has_keyword": int(bool(feat["has_keyword"])),
                }
            )


def print_explanations(explain: Dict[str, dict], top_n: int) -> None:
    if top_n <= 0:
        return
    flagged = [(uid, info) for uid, info in explain.items() if info["predicted"]]
    flagged.sort(key=lambda x: (x[1]["confidence"] != "high", -x[1]["total_points"], x[0]))
    print(f"Top {min(top_n, len(flagged))} flagged users with reasons:")
    for uid, info in flagged[:top_n]:
        reason_str = ",".join(info["reasons"]) if info["reasons"] else "NONE"
        print(f"  {uid} confidence={info['confidence']} points={info['total_points']} reasons={reason_str}")


def main() -> None:
    args = parse_args()

    input_json = Path(args.input_json)
    output_txt = Path(args.output_txt)
    bots_txt = Path(args.bots_txt) if args.bots_txt else None

    users, posts = load_dataset(input_json)
    features = build_user_features(users, posts)

    labeled_pairs: List[Tuple[Dict[str, dict], Set[str]]] = []
    if args.tune_dataset:
        for dataset_json, dataset_bots in args.tune_dataset:
            u, p = load_dataset(Path(dataset_json))
            f = build_user_features(u, p)
            b = load_bot_ids(Path(dataset_bots))
            labeled_pairs.append((f, b))

    if args.mode == "legacy":
        params_legacy = LegacyParams(z_thr=2.2, tweet_thr=60, dup_thr=2, hashtag_thr=0.30, cv_thr=1.3, avg_len_thr=95.0, score_thr=4)
        if not args.no_tune and labeled_pairs:
            params_legacy = tune_legacy_params(labeled_pairs, objective=args.calibration_objective)
            print(
                "Selected legacy params:",
                f"z_thr={params_legacy.z_thr}",
                f"tweet_thr={params_legacy.tweet_thr}",
                f"dup_thr={params_legacy.dup_thr}",
                f"hashtag_thr={params_legacy.hashtag_thr}",
                f"cv_thr={params_legacy.cv_thr}",
                f"avg_len_thr={params_legacy.avg_len_thr}",
                f"score_thr={params_legacy.score_thr}",
            )
            for idx, (f, b) in enumerate(labeled_pairs, start=1):
                preds = predict_bots_legacy(f, params_legacy)
                ev = evaluate_predictions(preds, b, f.keys())
                print_eval(f"tune_set_{idx}", ev, len(preds))

        predictions = predict_bots_legacy(features, params_legacy)
        explain = {uid: {"predicted": uid in predictions, "confidence": "legacy", "reasons": ["LEGACY_MODE"], "total_points": 0} for uid in features.keys()}
        stats = compute_dataset_stats(features)
        normalized = add_normalized_views(features, stats)
    else:
        params_adaptive = AdaptiveParams(
            tweet_rank_thr=0.90,
            z_rank_thr=0.90,
            dup_rank_thr=0.93,
            dupfrac_rank_thr=0.90,
            hashtag_rank_thr=0.85,
            mention_rank_thr=0.80,
            temporal_rank_thr=0.90,
            avg_len_rank_thr=0.80,
            short_inv_rank_thr=0.75,
            near_dup_frac_thr=0.20,
            high_score_thr=8,
            medium_score_thr=7,
        )
        if not args.no_tune and labeled_pairs:
            params_adaptive = tune_adaptive_params(
                labeled_pairs,
                objective=args.calibration_objective,
                min_posts_for_timing=args.min_posts_for_timing,
                min_posts_for_repetition=args.min_posts_for_repetition,
            )
            print(
                "Selected adaptive params:",
                f"tweet_rank_thr={params_adaptive.tweet_rank_thr}",
                f"z_rank_thr={params_adaptive.z_rank_thr}",
                f"dup_rank_thr={params_adaptive.dup_rank_thr}",
                f"dupfrac_rank_thr={params_adaptive.dupfrac_rank_thr}",
                f"hashtag_rank_thr={params_adaptive.hashtag_rank_thr}",
                f"mention_rank_thr={params_adaptive.mention_rank_thr}",
                f"temporal_rank_thr={params_adaptive.temporal_rank_thr}",
                f"avg_len_rank_thr={params_adaptive.avg_len_rank_thr}",
                f"short_inv_rank_thr={params_adaptive.short_inv_rank_thr}",
                f"near_dup_frac_thr={params_adaptive.near_dup_frac_thr}",
                f"high_score_thr={params_adaptive.high_score_thr}",
                f"medium_score_thr={params_adaptive.medium_score_thr}",
            )
            for idx, (f, b) in enumerate(labeled_pairs, start=1):
                preds, _ex, _st, _norm = predict_bots_adaptive(
                    f,
                    params_adaptive,
                    min_posts_for_timing=args.min_posts_for_timing,
                    min_posts_for_repetition=args.min_posts_for_repetition,
                )
                ev = evaluate_predictions(preds, b, f.keys())
                print_eval(f"tune_set_{idx}", ev, len(preds))

        predictions, explain, normalized_stats, normalized = predict_bots_adaptive(
            features,
            params_adaptive,
            min_posts_for_timing=args.min_posts_for_timing,
            min_posts_for_repetition=args.min_posts_for_repetition,
        )

        if bots_txt is not None and args.bootstrap_samples > 0:
            actual_bots = load_bot_ids(bots_txt)
            bootstrap_eval(
                features,
                actual_bots,
                params_adaptive,
                min_posts_for_timing=args.min_posts_for_timing,
                min_posts_for_repetition=args.min_posts_for_repetition,
                samples=args.bootstrap_samples,
                frac=args.bootstrap_frac,
                seed=args.bootstrap_seed,
            )

    output_txt.write_text("\n".join(sorted(predictions)) + ("\n" if predictions else ""), encoding="utf-8")
    print(f"Wrote {len(predictions)} predicted bot IDs to {output_txt}")

    if bots_txt is not None:
        actual_bots = load_bot_ids(bots_txt)
        ev = evaluate_predictions(predictions, actual_bots, features.keys())
        print_eval("input_dataset", ev, len(predictions))

    if args.explain_top > 0:
        print_explanations(explain, args.explain_top)

    if args.diagnostics_csv:
        diagnostics_path = Path(args.diagnostics_csv)
        write_diagnostics_csv(diagnostics_path, explain, normalized)
        print(f"Wrote diagnostics CSV to {diagnostics_path}")


if __name__ == "__main__":
    main()
