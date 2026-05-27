"""Metrics over the per-round posting log."""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .environment import _stance_to_score
from .agents import INITIAL_STANCES


def _posts_by_round(log: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    for row in log:
        out.setdefault(row["round"], []).append(row)
    return out


def opinion_convergence(log: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Per round: fraction of agents that share the top opinion."""
    out = []
    for r, posts in sorted(_posts_by_round(log).items()):
        counts = Counter(p["opinion"] for p in posts)
        top_op, top_n = counts.most_common(1)[0]
        out.append({
            "round": r,
            "majority_opinion": top_op,
            "majority_fraction": top_n / max(1, len(posts)),
            "num_unique_opinions": len(counts),
        })
    return out


def persona_retention(log: list[dict[str, Any]], agent_personas: dict[str, str],
                      embedder: Any = None) -> list[dict[str, float]]:
    """Per round mean cosine similarity between an agent's post and a
    one-line persona descriptor.

    If `embedder` is given (OllamaEmbedder), uses embeddings; else TF-IDF.
    Higher = post still reflects persona language.
    """
    out: list[dict[str, float]] = []
    by_round = _posts_by_round(log)
    if not by_round:
        return out

    persona_texts = list(agent_personas.values())
    if not persona_texts:
        return out

    for r in sorted(by_round.keys()):
        posts = by_round[r]
        agent_ids = [p["agent_id"] for p in posts]
        post_texts = [p["text"] for p in posts]
        descriptor_texts = [agent_personas.get(aid, "") for aid in agent_ids]

        if embedder is not None:
            try:
                P = embedder.embed_many(post_texts)
                D = embedder.embed_many(descriptor_texts)
                sims = [float(np.dot(P[i], D[i])) for i in range(len(post_texts))]
                mean_sim = float(np.mean(sims)) if sims else 0.0
            except Exception:  # noqa: BLE001
                mean_sim = 0.0
        else:
            try:
                vec = TfidfVectorizer(min_df=1, lowercase=True).fit(post_texts + descriptor_texts)
                P = vec.transform(post_texts)
                D = vec.transform(descriptor_texts)
                sims = []
                for i in range(P.shape[0]):
                    sims.append(float(cosine_similarity(P[i], D[i])[0, 0]))
                mean_sim = float(np.mean(sims)) if sims else 0.0
            except ValueError:
                mean_sim = 0.0
        out.append({"round": r, "persona_retention": mean_sim})
    return out


def opinion_shift_rate(log: list[dict[str, Any]],
                       initial_opinions: dict[str, str]) -> list[dict[str, float]]:
    """Per round: fraction of agents whose opinion differs from their initial."""
    out = []
    for r, posts in sorted(_posts_by_round(log).items()):
        shifts = sum(1 for p in posts if p["opinion"] != initial_opinions.get(p["agent_id"]))
        out.append({"round": r, "shift_rate": shifts / max(1, len(posts))})
    return out


def confidence_trajectory(log: list[dict[str, Any]]) -> list[dict[str, float]]:
    out = []
    for r, posts in sorted(_posts_by_round(log).items()):
        confs = [float(p["confidence"]) for p in posts]
        out.append({
            "round": r,
            "mean_confidence": float(np.mean(confs)) if confs else 0.0,
            "std_confidence": float(np.std(confs)) if confs else 0.0,
        })
    return out


def minority_survival(log: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare round 0 vs final round: how many opinions present at start still survive?"""
    by_round = _posts_by_round(log)
    if not by_round:
        return {"survived": 0, "initial_unique": 0, "survival_rate": 0.0}
    first = min(by_round.keys())
    last = max(by_round.keys())
    initial_ops = set(p["opinion"] for p in by_round[first])
    final_ops = set(p["opinion"] for p in by_round[last])
    survived = initial_ops & final_ops
    # Specifically how many *minority* opinions survived (excluding initial majority)
    init_counts = Counter(p["opinion"] for p in by_round[first])
    initial_majority, _ = init_counts.most_common(1)[0]
    minority_initial = initial_ops - {initial_majority}
    minority_survived = minority_initial & final_ops
    return {
        "initial_unique": len(initial_ops),
        "final_unique": len(final_ops),
        "survived_any": len(survived),
        "minority_initial": len(minority_initial),
        "minority_survived": len(minority_survived),
        "minority_survival_rate": (
            len(minority_survived) / len(minority_initial)
            if minority_initial else 1.0
        ),
    }


def language_similarity(log: list[dict[str, Any]],
                        embedder: Any = None) -> list[dict[str, float]]:
    """Per round: mean pairwise cosine similarity between posts.

    If `embedder` is given, uses semantic embeddings; else TF-IDF.
    Higher = posts look more alike (less linguistic diversity).
    """
    out: list[dict[str, float]] = []
    by_round = _posts_by_round(log)
    if not by_round:
        return out
    all_texts = [p["text"] for r in sorted(by_round.keys()) for p in by_round[r]]
    if len(all_texts) < 2:
        return out

    vec = None
    if embedder is None:
        try:
            vec = TfidfVectorizer(min_df=1, lowercase=True).fit(all_texts)
        except ValueError:
            return out

    for r in sorted(by_round.keys()):
        texts = [p["text"] for p in by_round[r]]
        if len(texts) < 2:
            out.append({"round": r, "mean_pairwise_sim": 0.0})
            continue
        try:
            if embedder is not None:
                M = embedder.embed_many(texts)
                S = M @ M.T  # embeddings are L2-normalized → dot product = cosine
            else:
                M = vec.transform(texts)
                S = cosine_similarity(M)
            n = S.shape[0]
            iu = np.triu_indices(n, k=1)
            mean_sim = float(np.asarray(S)[iu].mean()) if iu[0].size else 0.0
        except (ValueError, Exception):  # noqa: BLE001
            mean_sim = 0.0
        out.append({"round": r, "mean_pairwise_sim": mean_sim})
    return out


def confidence_inflation(conf_traj: list[dict[str, float]],
                         conv_traj: list[dict[str, float]]) -> dict[str, float]:
    """Scalar summary: does mean confidence rise as majority fraction rises?"""
    if len(conf_traj) < 2:
        return {"delta_confidence": 0.0, "delta_majority_fraction": 0.0}
    return {
        "delta_confidence": conf_traj[-1]["mean_confidence"] - conf_traj[0]["mean_confidence"],
        "delta_majority_fraction": (
            conv_traj[-1]["majority_fraction"] - conv_traj[0]["majority_fraction"]
        ),
    }


def stratified_shift_by_group(log: list[dict[str, Any]],
                              initial_opinions: dict[str, str],
                              agent_groups: dict[str, str]) -> list[dict[str, Any]]:
    """EXP-7: per-round shift_rate stratified by persona group."""
    out: list[dict[str, Any]] = []
    for r, posts in sorted(_posts_by_round(log).items()):
        by_group: dict[str, list[int]] = {}
        for p in posts:
            grp = agent_groups.get(p["agent_id"], "unknown")
            shifted = 1 if p["opinion"] != initial_opinions.get(p["agent_id"]) else 0
            by_group.setdefault(grp, []).append(shifted)
        row: dict[str, Any] = {"round": r}
        for grp, vals in by_group.items():
            row[f"shift_rate_{grp}"] = sum(vals) / len(vals)
            row[f"n_{grp}"] = len(vals)
        out.append(row)
    return out


def stratified_retention_by_group(log: list[dict[str, Any]],
                                  agent_personas: dict[str, str],
                                  agent_groups: dict[str, str],
                                  embedder: Any = None) -> list[dict[str, Any]]:
    """EXP-7: per-round persona retention stratified by group."""
    out: list[dict[str, Any]] = []
    by_round = _posts_by_round(log)
    for r in sorted(by_round.keys()):
        posts = by_round[r]
        by_group: dict[str, list[float]] = {}
        if embedder is not None:
            try:
                for p in posts:
                    grp = agent_groups.get(p["agent_id"], "unknown")
                    v_post = embedder.embed(p["text"])
                    v_pers = embedder.embed(agent_personas.get(p["agent_id"], ""))
                    by_group.setdefault(grp, []).append(float(np.dot(v_post, v_pers)))
            except Exception:  # noqa: BLE001
                pass
        else:
            try:
                all_texts = [p["text"] for p in posts] + \
                            [agent_personas.get(p["agent_id"], "") for p in posts]
                vec = TfidfVectorizer(min_df=1, lowercase=True).fit(all_texts)
                for i, p in enumerate(posts):
                    grp = agent_groups.get(p["agent_id"], "unknown")
                    pv = vec.transform([p["text"]])
                    dv = vec.transform([agent_personas.get(p["agent_id"], "")])
                    by_group.setdefault(grp, []).append(
                        float(cosine_similarity(pv, dv)[0, 0])
                    )
            except ValueError:
                pass
        row: dict[str, Any] = {"round": r}
        for grp, vals in by_group.items():
            row[f"retention_{grp}"] = float(np.mean(vals)) if vals else 0.0
        out.append(row)
    return out


def summarize(log: list[dict[str, Any]],
              initial_opinions: dict[str, str],
              agent_personas: dict[str, str],
              agent_groups: dict[str, str] | None = None,
              embedder: Any = None) -> dict[str, Any]:
    conv = opinion_convergence(log)
    pers = persona_retention(log, agent_personas, embedder=embedder)
    shift = opinion_shift_rate(log, initial_opinions)
    conf = confidence_trajectory(log)
    sim = language_similarity(log, embedder=embedder)
    minority = minority_survival(log)
    inflation = confidence_inflation(conf, conv)
    strat_shift = (
        stratified_shift_by_group(log, initial_opinions, agent_groups)
        if agent_groups else []
    )
    strat_ret = (
        stratified_retention_by_group(log, agent_personas, agent_groups,
                                       embedder=embedder)
        if agent_groups else []
    )

    final_majority_fraction = conv[-1]["majority_fraction"] if conv else 0.0
    final_persona_retention = pers[-1]["persona_retention"] if pers else 0.0
    final_pairwise_sim = sim[-1]["mean_pairwise_sim"] if sim else 0.0

    return {
        "per_round": {
            "convergence": conv,
            "persona_retention": pers,
            "shift_rate": shift,
            "confidence": conf,
            "language_similarity": sim,
            "shift_rate_by_group": strat_shift,
            "retention_by_group": strat_ret,
        },
        "scalar": {
            "final_majority_fraction": final_majority_fraction,
            "final_persona_retention": final_persona_retention,
            "final_mean_pairwise_sim": final_pairwise_sim,
            **minority,
            **inflation,
            # Operational definition: "persona collapse score"
            # Higher = worse: high consensus + high linguistic similarity + low persona retention
            "persona_collapse_score": round(
                0.5 * final_majority_fraction
                + 0.3 * final_pairwise_sim
                + 0.2 * (1.0 - final_persona_retention),
                4,
            ),
        },
    }
