from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
from typing import Any, Iterable

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# ── Paths ──────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Beginner Fitness Coach", page_icon="💪", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "exercises.json"
IMAGE_ROOT = BASE_DIR / "data" / "exercises"
CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)


# ── Data loading ───────────────────────────────────────────────────────────────

def _safe_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _safe_str(value: Any, default: str = "unknown") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def build_search_text(row: dict[str, Any]) -> str:
    primary = ", ".join(_safe_list(row.get("primaryMuscles"))) or "none"
    secondary = ", ".join(_safe_list(row.get("secondaryMuscles"))) or "none"
    instructions = " ".join(_safe_list(row.get("instructions"))) or "none"
    return (
        f"Exercise: {_safe_str(row.get('name'))}\n"
        f"Force: {_safe_str(row.get('force'))}\n"
        f"Level: {_safe_str(row.get('level'))}\n"
        f"Mechanic: {_safe_str(row.get('mechanic'))}\n"
        f"Equipment: {_safe_str(row.get('equipment'))}\n"
        f"Category: {_safe_str(row.get('category'))}\n"
        f"Primary muscles: {primary}\n"
        f"Secondary muscles: {secondary}\n"
        f"Instructions: {instructions}"
    )


def load_exercises_dataframe(json_path: str | Path) -> pd.DataFrame:
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    cleaned_records = []
    for row in records:
        cleaned = {
            "id": _safe_str(row.get("id"), default="unknown_id"),
            "name": _safe_str(row.get("name")),
            "force": _safe_str(row.get("force")),
            "level": _safe_str(row.get("level")),
            "mechanic": _safe_str(row.get("mechanic")),
            "equipment": _safe_str(row.get("equipment")),
            "category": _safe_str(row.get("category")),
            "primaryMuscles": _safe_list(row.get("primaryMuscles")),
            "secondaryMuscles": _safe_list(row.get("secondaryMuscles")),
            "instructions": _safe_list(row.get("instructions")),
            "images": _safe_list(row.get("images")),
        }
        cleaned["all_muscles"] = sorted(set(cleaned["primaryMuscles"] + cleaned["secondaryMuscles"]))
        cleaned["search_text"] = build_search_text(cleaned)
        cleaned_records.append(cleaned)

    return pd.DataFrame(cleaned_records)


# ── Retrieval ──────────────────────────────────────────────────────────────────

def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return scores
    min_s, max_s = scores.min(), scores.max()
    if np.isclose(min_s, max_s):
        return np.ones_like(scores)
    return (scores - min_s) / (max_s - min_s)


def lexical_retrieval(df: pd.DataFrame, user_query: str, top_k: int = 8) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    corpus = (
        df["name"].fillna("")
        + " "
        + df["search_text"].fillna("")
        + " "
        + df["instructions"].apply(
            lambda steps: " ".join(steps) if isinstance(steps, list) else ""
        )
    ).tolist()

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    doc_matrix = vectorizer.fit_transform(corpus)
    query_vector = vectorizer.transform([user_query])
    scores = cosine_similarity(query_vector, doc_matrix).flatten()
    ranked_idx = np.argsort(scores)[::-1][:top_k]

    results = df.iloc[ranked_idx].copy()
    results["keyword_score"] = scores[ranked_idx]
    return results.reset_index(drop=True)


def _cache_key_for_ids(ids: Iterable[str]) -> str:
    return hashlib.md5("|".join(sorted(ids)).encode("utf-8")).hexdigest()


def semantic_retrieval(df: pd.DataFrame, user_query: str, top_k: int = 8) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    if SentenceTransformer is None:
        fallback = lexical_retrieval(df, user_query, top_k=top_k)
        fallback["semantic_score"] = fallback.get("keyword_score", 0.0)
        return fallback

    model_name = "all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)

    ids = df["id"].astype(str).tolist()
    cache_key = _cache_key_for_ids(ids)
    emb_path = CACHE_DIR / f"semantic_embeddings_{cache_key}.npy"
    meta_path = CACHE_DIR / f"semantic_embeddings_{cache_key}.json"

    if emb_path.exists() and meta_path.exists():
        embeddings = np.load(emb_path)
    else:
        embeddings = model.encode(
            df["search_text"].tolist(),
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        np.save(emb_path, embeddings)
        meta_path.write_text(
            json.dumps({"ids": ids, "model": model_name}, indent=2), encoding="utf-8"
        )

    query_embedding = model.encode([user_query], show_progress_bar=False, normalize_embeddings=True)
    scores = cosine_similarity(query_embedding, embeddings).flatten()
    ranked_idx = np.argsort(scores)[::-1][:top_k]

    results = df.iloc[ranked_idx].copy()
    results["semantic_score"] = scores[ranked_idx]
    return results.reset_index(drop=True)


def run_hybrid_retrieval(
    df: pd.DataFrame,
    user_query: str,
    top_k: int = 8,
    keyword_weight: float = 0.5,
    semantic_weight: float = 0.5,
) -> pd.DataFrame:
    if df.empty:
        return df

    lexical_df = lexical_retrieval(df, user_query, top_k=min(top_k * 2, len(df)))
    semantic_df = semantic_retrieval(df, user_query, top_k=min(top_k * 2, len(df)))

    merged = df.copy()
    merged = merged.merge(lexical_df[["id", "keyword_score"]], on="id", how="left")
    merged = merged.merge(semantic_df[["id", "semantic_score"]], on="id", how="left")
    merged["keyword_score"] = merged["keyword_score"].fillna(0.0)
    merged["semantic_score"] = merged["semantic_score"].fillna(0.0)
    merged["keyword_norm"] = _normalize_scores(merged["keyword_score"].values)
    merged["semantic_norm"] = _normalize_scores(merged["semantic_score"].values)
    merged["hybrid_score"] = (
        keyword_weight * merged["keyword_norm"] + semantic_weight * merged["semantic_norm"]
    )

    return merged.sort_values("hybrid_score", ascending=False).head(top_k).reset_index(drop=True)


# ── Generation ─────────────────────────────────────────────────────────────────

def build_context_from_retrieved(retrieved_df: pd.DataFrame) -> str:
    if retrieved_df.empty:
        return "No exercises were retrieved."

    chunks = []
    for _, row in retrieved_df.iterrows():
        primary = ", ".join(row.get("primaryMuscles", [])) or "none"
        secondary = ", ".join(row.get("secondaryMuscles", [])) or "none"
        instructions = " ".join(row.get("instructions", []))
        chunks.append(
            f"Exercise: {row['name']}\n"
            f"Level: {row['level']}\n"
            f"Equipment: {row['equipment']}\n"
            f"Category: {row['category']}\n"
            f"Primary muscles: {primary}\n"
            f"Secondary muscles: {secondary}\n"
            f"Instructions: {instructions}"
        )
    return "\n\n".join(chunks)


def _call_openai(prompt: str) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    client = OpenAI(api_key=api_key)
    response = client.responses.create(model="gpt-4.1-mini", input=prompt)
    return response.output_text.strip()


def _pick_section(rows: list[dict], keyword: str, n: int) -> list[dict]:
    matches = [r for r in rows if keyword.lower() in r.get("category", "").lower()]
    return matches[:n] if len(matches) >= n else rows[:n]


def _format_sets_reps(duration_hint: str, category: str) -> str:
    if "stretch" in category.lower():
        return "Hold 20-30 seconds"
    return "3 sets of 10-12 reps"


def _rule_based_plan(user_query: str, retrieved_df: pd.DataFrame) -> str:
    if retrieved_df.empty:
        return (
            "## No workout could be created\n\n"
            "No matching exercises were found. Try describing your workout differently."
        )

    rows = retrieved_df.to_dict(orient="records")
    warmup = _pick_section(rows, "stretch", 2)
    cooldown = _pick_section(list(reversed(rows)), "stretch", 2)
    used_names = {x["name"] for x in warmup + cooldown}
    main_workout = [r for r in rows if r["name"] not in used_names][: max(3, min(6, len(rows)))]

    lines = [
        "## Your Workout Plan",
        "",
        f"**Based on your request:** {user_query}",
        "",
        "### Warm-up",
    ]
    for item in warmup:
        lines.append(
            f"- **{item['name']}** — 1-2 rounds. {_format_sets_reps('', item['category'])}. "
            "Prepares the body for movement."
        )
    lines += ["", "### Main Workout"]
    for item in main_workout:
        primary = ", ".join(item.get("primaryMuscles", [])) or "general fitness"
        lines.append(
            f"- **{item['name']}** — {_format_sets_reps('', item['category'])}. "
            f"Targets: {primary}."
        )
    lines += ["", "### Cooldown"]
    for item in cooldown:
        lines.append(f"- **{item['name']}** — Hold 20-30 seconds.")
    lines += ["", "### Substitutions"]
    subs = main_workout[:2] if main_workout else rows[:2]
    for item in subs:
        alt = retrieved_df[retrieved_df["category"] == item["category"]]
        alt_name = next(
            (r["name"] for _, r in alt.iterrows() if r["name"] != item["name"]), None
        )
        if alt_name:
            lines.append(f"- If **{item['name']}** feels too hard, try **{alt_name}** instead.")
        else:
            lines.append(f"- If **{item['name']}** feels too hard, reduce the reps.")
    lines += [
        "",
        "### Notes",
        "- All exercises are drawn from the retrieved dataset.",
        "- Stop if anything feels painful and prioritize good form.",
    ]
    return "\n".join(lines)

def generate_workout_plan(user_query: str, retrieved_df: pd.DataFrame) -> str:
    context = build_context_from_retrieved(retrieved_df)
    prompt = f"""
You are a beginner fitness coach. Only use the retrieved exercises provided below.
Do not invent any exercises that are not in the context.
Keep the plan beginner-friendly and practical.
Give instructions for each exercise in the workout.

User request: {user_query}

Retrieved exercises:
{context}

Return the answer in markdown with these sections:
1. Workout title
2. Warm-up
3. Main workout
4. Cooldown
5. Substitutions
6. Notes

Include short explanations for why each exercise was chosen.
""".strip()

    llm_output = _call_openai(prompt)
    if llm_output:
        return llm_output

    return _rule_based_plan(user_query, retrieved_df)


# ── Streamlit UI ───────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data():
    return load_exercises_dataframe(DATA_PATH)

def main():
    st.title("Beginner Fitness Coach")
    st.write("Describe the workout you want and the app will build a plan for you. The plan will include a warm-up, main workout, and cooldown.")

    df = load_data()

    user_query = st.text_area(
        "What kind of workout are you looking for?",
        value=(
            "I am a beginner and want a 30 minute full body workout "
            "with bodyweight only."
        ),
        height=120,
    )

    #top_k = st.slider("Number of exercises to retrieve", min_value=1, max_value=5, value=3)

    if st.button("Generate Workout", type="primary"):
        with st.spinner("Retrieving exercises and building your workout..."):
            retrieved = run_hybrid_retrieval(df=df, user_query=user_query)
            plan = generate_workout_plan(user_query=user_query, retrieved_df=retrieved)

        st.subheader("Generated workout plan")
        st.markdown(plan)


if __name__ == "__main__":
    main()