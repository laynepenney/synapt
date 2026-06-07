"""CodeMemo benchmark evaluation for synapt recall.

A coding-assistant memory benchmark: ingest multi-session Claude Code
transcripts, retrieve context for coding questions, generate + judge answers.

Usage:
    # Full run on one project (requires OPENAI_API_KEY):
    python evaluation/codememo/eval.py --project project_01_cli_tool

    # RecallDB mode (FTS5 + embeddings):
    python evaluation/codememo/eval.py --project project_01_cli_tool --recalldb

    # Full pipeline (RecallDB + enrich + consolidate + knowledge graph):
    python evaluation/codememo/eval.py --project project_01_cli_tool --full-pipeline

    # Retrieval-only (no API key needed):
    python evaluation/codememo/eval.py --retrieval-only

    # All projects:
    python evaluation/codememo/eval.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, runtime_checkable

try:
    from evaluation.codememo.schema import CATEGORY_NAMES, ALL_CATEGORIES
except ModuleNotFoundError:
    # Tests may load this module by file path, where evaluation/ is not a package.
    _SCHEMA_PATH = Path(__file__).with_name("schema.py")
    _SCHEMA_SPEC = importlib.util.spec_from_file_location("codememo_schema", _SCHEMA_PATH)
    assert _SCHEMA_SPEC is not None
    assert _SCHEMA_SPEC.loader is not None
    _schema_module = importlib.util.module_from_spec(_SCHEMA_SPEC)
    sys.modules[_SCHEMA_SPEC.name] = _schema_module
    _SCHEMA_SPEC.loader.exec_module(_schema_module)
    CATEGORY_NAMES = _schema_module.CATEGORY_NAMES
    ALL_CATEGORIES = _schema_module.ALL_CATEGORIES

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------

CODEMEMO_DIR = Path(__file__).parent
_LOCAL_DATA = CODEMEMO_DIR / "data"

# HuggingFace dataset ID — auto-downloaded if no local data/ directory exists
HF_DATASET_ID = "laynepro/codememo-benchmark"
_SMALL_SAMPLE_THRESHOLD = 10
_NEAR_CEILING_J_SCORE = 95.0
_RECALL_REASONING_GAP_THRESHOLD = 20.0
_NEGATIVE_EVIDENCE_RE = re.compile(
    r"\b("
    r"never|no |none|not |did not|didn't|was not|wasn't|were not|weren't|"
    r"not implemented|not added|not called|not used|no evidence|no such"
    r")\b",
    re.IGNORECASE,
)


def _resolve_data_dir() -> Path:
    """Return data directory, downloading from HuggingFace if needed."""
    if _LOCAL_DATA.is_dir() and any(_LOCAL_DATA.iterdir()):
        return _LOCAL_DATA
    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(HF_DATASET_ID, repo_type="dataset")
        return Path(path)
    except ImportError:
        return _LOCAL_DATA  # fall back, will error later if empty


DATA_DIR: Path | None = None


def _get_data_dir() -> Path:
    """Resolve CodeMemo data lazily so importing this module is side-effect free."""
    global DATA_DIR
    if DATA_DIR is None:
        DATA_DIR = _resolve_data_dir()
    return DATA_DIR


# ---------------------------------------------------------------------------
# Provenance + audit
# ---------------------------------------------------------------------------

_AUDIT_FILE = CODEMEMO_DIR / "audit.jsonl"


def _get_provenance() -> dict:
    """Capture reproducibility metadata for this run."""
    prov: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable,
        "python_version": sys.version.split()[0],
    }
    # Synapt version + code ref
    try:
        import synapt.recall
        prov["synapt_version"] = synapt.recall.__version__
    except ImportError:
        prov["synapt_version"] = "not installed"
    # Git ref from synapt repo
    synapt_repo = Path(__file__).resolve().parents[2]
    try:
        prov["code_ref"] = subprocess.run(
            ["git", "-C", str(synapt_repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        prov["code_dirty"] = bool(subprocess.run(
            ["git", "-C", str(synapt_repo), "diff", "--quiet"],
            capture_output=True, timeout=5,
        ).returncode)
    except Exception:
        prov["code_ref"] = os.environ.get("SYNAPT_CODE_REF", "unknown")
    # Relevant env vars
    env_keys = [k for k in os.environ if k.startswith("SYNAPT_")]
    if env_keys:
        prov["env_overrides"] = {k: os.environ[k] for k in sorted(env_keys)}
    return prov


def _audit_log(entry: dict) -> None:
    """Append a JSON entry to audit.jsonl."""
    try:
        _AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_AUDIT_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"  [audit] Warning: could not write audit log: {e}")


def _audit_start(pipeline: str, model: str, provenance: dict) -> str:
    """Log a RUNNING entry and return its timestamp (used as ID)."""
    ts = provenance["timestamp"]
    entry = {
        "type": "eval",
        "eval_type": "codememo",
        "timestamp": ts,
        "outcome": "RUNNING",
        "pipeline": pipeline,
        "model": model,
        "code_ref": provenance.get("code_ref", "unknown"),
        "synapt_version": provenance.get("synapt_version", "unknown"),
        "env_overrides": provenance.get("env_overrides", {}),
    }
    _audit_log(entry)
    print(f"  [audit] Logged RUNNING (ref={provenance.get('code_ref', '?')}, "
          f"v={provenance.get('synapt_version', '?')})")
    return ts


def _audit_finish(start_ts: str, summary: dict, outcome: str = "SUCCESS") -> None:
    """Log a completion entry with scores."""
    entry = {
        "type": "eval",
        "eval_type": "codememo",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "start_timestamp": start_ts,
        "outcome": outcome,
        "pipeline": summary.get("pipeline_mode", "unknown"),
        "model": summary.get("model", "unknown"),
        "questions_evaluated": summary.get("questions_evaluated", 0),
        "j_score_overall": summary.get("j_score_overall"),
        "j_score_factual": summary.get("j_score_factual"),
        "j_score_debug": summary.get("j_score_debug"),
        "j_score_architecture": summary.get("j_score_architecture"),
        "j_score_temporal": summary.get("j_score_temporal"),
        "j_score_convention": summary.get("j_score_convention"),
        "j_score_cross-session": summary.get("j_score_cross-session"),
        "f1_overall": summary.get("f1_overall"),
    }
    _audit_log(entry)
    j = summary.get("j_score_overall", "?")
    print(f"  [audit] Logged {outcome} — J-Score: {j}%")


def _add_validity_metadata(summary: dict) -> dict:
    """Attach benchmark-validity diagnostics to a CodeMemo summary."""
    category_diagnostics: dict[str, dict] = {}
    validity_notes: list[str] = []
    near_ceiling: list[tuple[str, float, int]] = []
    reasoning_gap: list[tuple[str, float, float]] = []

    max_chunks = summary.get("max_chunks", 20)
    for cat in sorted(ALL_CATEGORIES):
        name = CATEGORY_NAMES[cat]
        n = summary.get(f"n_{name}")
        j_score = summary.get(f"j_score_{name}")
        recall = summary.get(f"recall@{max_chunks}_{name}")

        if n is None and j_score is None and recall is None:
            continue

        entry: dict[str, int | float | bool] = {}
        if n is not None:
            entry["questions"] = n
            entry["small_sample"] = n < _SMALL_SAMPLE_THRESHOLD
        if j_score is not None:
            entry["j_score"] = j_score
            entry["headroom"] = round(max(0.0, 100.0 - j_score), 2)
            entry["near_ceiling"] = j_score >= _NEAR_CEILING_J_SCORE
        if recall is not None:
            entry["recall_at_k"] = recall
        if j_score is not None and recall is not None:
            gap = round(j_score - recall, 2)
            entry["reasoning_gap"] = gap
            entry["high_reasoning_gap"] = gap >= _RECALL_REASONING_GAP_THRESHOLD
        category_diagnostics[name] = entry

        if (
            n is not None
            and j_score is not None
            and j_score >= _NEAR_CEILING_J_SCORE
        ):
            near_ceiling.append((name, j_score, n))
        if (
            j_score is not None
            and recall is not None
            and j_score >= _NEAR_CEILING_J_SCORE
            and (j_score - recall) >= _RECALL_REASONING_GAP_THRESHOLD
        ):
            reasoning_gap.append((name, j_score, recall))

    if len(summary.get("projects", [])) == 1:
        validity_notes.append(
            "Single-project slice; prefer all-project CodeMemo runs for headline claims."
        )
    for name, j_score, n in near_ceiling:
        validity_notes.append(
            f"Category '{name}' is near ceiling on this slice ({j_score:.2f}% over {n} questions)."
        )
    for name, j_score, recall in reasoning_gap:
        validity_notes.append(
            f"Category '{name}' has near-ceiling J-score ({j_score:.2f}%) despite lower retrieval recall ({recall:.2f}%), so answer-model or judge effects may be masking retrieval weakness."
        )

    if category_diagnostics:
        summary["category_diagnostics"] = category_diagnostics
    if validity_notes:
        summary["validity_notes"] = validity_notes
    return summary


# ---------------------------------------------------------------------------
# SystemUnderTest protocol — plug in any memory system
# ---------------------------------------------------------------------------

@runtime_checkable
class SystemUnderTest(Protocol):
    """Interface for a memory system that can be benchmarked."""

    def ingest(self, session_paths: list[Path]) -> None:
        """Ingest one project's session transcripts."""
        ...

    def query(self, question: str, max_chunks: int = 20) -> str:
        """Retrieve context for a question. Returns formatted text."""
        ...


# ---------------------------------------------------------------------------
# SynaptSUT — synapt recall implementation
# ---------------------------------------------------------------------------

class SynaptSUT:
    """SystemUnderTest backed by synapt recall."""

    def __init__(
        self,
        mode: str = "baseline",
        enrich_model: str = "",
        max_knowledge: int | None = None,
        knowledge_boost: float | None = None,
        work_dir: str = "",
    ):
        self.mode = mode  # "baseline", "recalldb", "full-pipeline"
        self.enrich_model = enrich_model
        self.max_knowledge = max_knowledge
        self.knowledge_boost = knowledge_boost
        self._index = None
        self._db = None
        self._work_dir: Path | None = None
        self._persistent_work_dir = work_dir  # if set, reuse instead of tmpdir

    def _project_key(self, session_paths: list[Path]) -> str:
        """Derive a stable per-project work-dir key from session paths."""
        if not session_paths:
            return "project"
        sessions_dir = session_paths[0].parent
        project_dir = sessions_dir.parent
        return project_dir.name or sessions_dir.name or "project"

    def ingest(self, session_paths: list[Path]) -> None:
        """Build a synapt recall index from session transcripts."""
        from synapt.recall.core import build_index, parse_transcript
        from synapt.recall.storage import RecallDB

        if self._persistent_work_dir:
            project_key = self._project_key(session_paths)
            self._work_dir = Path(self._persistent_work_dir) / project_key
            self._work_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._work_dir = Path(tempfile.mkdtemp(prefix="codememo_"))
        index_dir = self._work_dir / "index"
        index_dir.mkdir(exist_ok=True)

        # Sessions are already in Claude Code JSONL format — no conversion
        # needed. Just point build_index at a directory containing them.
        transcript_dir = self._work_dir / "transcripts"
        shutil.rmtree(transcript_dir, ignore_errors=True)
        transcript_dir.mkdir()

        # Symlink sessions into the transcript dir
        for sp in session_paths:
            dst = transcript_dir / sp.name
            if not dst.exists():
                dst.symlink_to(sp.resolve())

        if self.mode in ("recalldb", "full-pipeline"):
            db = RecallDB(index_dir / "recall.db")
            all_chunks = []
            for sp in session_paths:
                chunks = parse_transcript(transcript_dir / sp.name)
                all_chunks.extend(chunks)
            db.save_chunks(all_chunks)
            self._db = db
        else:
            db = None

        if self.mode == "full-pipeline":
            self._run_enrichment(transcript_dir, index_dir)

        self._index = build_index(
            source_dir=transcript_dir,
            use_embeddings=True,
            cache_dir=index_dir,
            db=self._db,
        )

    def _run_enrichment(self, transcript_dir: Path, index_dir: Path) -> None:
        """Run enrich + consolidate for full pipeline mode."""
        from synapt.recall.enrich import enrich_all
        from synapt.recall.consolidate import consolidate
        from synapt.recall.knowledge import read_nodes
        from synapt.recall.journal import JournalEntry, append_entry

        # Set up directory structure that enrich_all expects
        project_dir = self._work_dir / "project"
        shutil.rmtree(project_dir, ignore_errors=True)
        project_dir.mkdir()
        data_dir = project_dir / ".synapt" / "recall"
        # _worktree_name(project_dir) resolves to project_dir.name = "project"
        wt_dir = data_dir / "worktrees" / "project"
        archive_dir = wt_dir / "transcripts"
        archive_dir.mkdir(parents=True)

        # Copy transcripts to archive
        for tp in sorted(transcript_dir.glob("*.jsonl")):
            dst = archive_dir / tp.name
            if not dst.exists():
                shutil.copy2(tp, dst)

        # Create journal stubs
        journal_path = wt_dir / "journal.jsonl"
        for tp in sorted(archive_dir.glob("*.jsonl")):
            session_id = tp.stem
            timestamp = ""
            try:
                with open(tp, encoding="utf-8") as tf:
                    first_line = tf.readline().strip()
                timestamp = json.loads(first_line).get("timestamp", "")
            except (json.JSONDecodeError, IndexError):
                pass
            stub = JournalEntry(
                timestamp=timestamp or "2025-01-01T00:00:00+00:00",
                session_id=session_id,
                auto=True,
                enriched=False,
            )
            append_entry(stub, journal_path)

        # Enrich
        enrich_kwargs = {"project_dir": project_dir, "max_entries": 0}
        if self.enrich_model:
            enrich_kwargs["model"] = self.enrich_model
        enriched_count = enrich_all(**enrich_kwargs)
        print(f"    Enriched {enriched_count} sessions")

        # Detect content profile for adaptive consolidation filters
        from synapt.recall.content_profile import detect_content_profile
        from synapt.recall.core import parse_transcript
        enrich_chunks = []
        for tp in sorted(archive_dir.glob("*.jsonl")):
            enrich_chunks.extend(parse_transcript(tp))
        c_profile = detect_content_profile(enrich_chunks)

        # Consolidate
        consolidate_kwargs = {
            "project_dir": project_dir,
            "force": True,
            "min_entries": 2,
            "content_profile": c_profile,
        }
        if self.enrich_model:
            consolidate_kwargs["model"] = self.enrich_model
        con_result = consolidate(**consolidate_kwargs)
        print(f"    Knowledge: {con_result.nodes_created} created, "
              f"{con_result.nodes_corroborated} corroborated, "
              f"{con_result.clusters_found} clusters")

        # Sync knowledge nodes into DB
        kn_path = data_dir / "knowledge.jsonl"
        if kn_path.exists() and self._db is not None:
            kn_nodes = read_nodes(kn_path)
            if kn_nodes:
                self._db.save_knowledge_nodes([n.to_dict() for n in kn_nodes])

        # Release MLX resources
        import gc
        try:
            import mlx.core as mx
            mx.metal.clear_cache()
        except Exception:
            pass
        gc.collect()

    def query(self, question: str, max_chunks: int = 20) -> str:
        """Retrieve context via synapt recall."""
        if self._index is None:
            raise RuntimeError("Must call ingest() before query()")
        return self._index.lookup(
            query=question,
            max_chunks=max_chunks,
            max_tokens=2000,
            half_life=0,
            threshold_ratio=0.0,
            depth="full",
            knowledge_boost=self.knowledge_boost,
            max_knowledge=self.max_knowledge,
        )

    def stats(self) -> dict:
        """Return index statistics."""
        if self._index is None:
            return {}
        return self._index.stats()

    def reset_working_memory(self) -> None:
        """Clear only session-scoped working memory between repeat passes."""
        wm = getattr(self._index, "_working_memory", None)
        if wm is not None and hasattr(wm, "clear"):
            wm.clear()

    def close(self) -> None:
        """Clean up resources."""
        if self._db is not None:
            try:
                self._db.close()
            except Exception:
                pass
        if self._work_dir and self._work_dir.exists() and not self._persistent_work_dir:
            shutil.rmtree(self._work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

ANSWER_PROMPT = """\
You are an intelligent coding assistant with access to memories from past \
development sessions. Your task is to answer questions about a software \
project based on retrieved session transcripts.

# CONTEXT:
You have access to conversation transcripts from coding sessions on a \
software project. These contain discussions about code changes, debugging \
sessions, architecture decisions, and development conventions.

# INSTRUCTIONS:
1. Carefully analyze ALL provided transcript excerpts
2. Pay attention to file paths, function names, config values, library \
versions, and error messages — be precise with technical details
3. If the question asks about a specific decision or change, look for the \
reasoning and outcome in the transcripts
4. For code-related answers, include the exact values (file paths, config \
keys, version numbers) from the transcripts
5. If transcripts contain contradictory information (e.g., a value was \
changed), report the most recent state unless the question asks about history
6. Normalize file paths — treat "./src/foo.py" and "src/foo.py" as the same
7. NEVER say "not mentioned" if any relevant detail appears in the context
8. Answer with specific technical details. Be precise and complete but \
concise — use 2-3 sentences if needed to cover all relevant details.

Retrieved context:
{context}

Question: {question}
Answer:"""

JUDGE_PROMPT = """\
Your task is to label an answer to a coding question as "CORRECT" or "WRONG".

You will be given: (1) a question about a software project's development \
history, (2) a gold (ground truth) answer, (3) a generated answer to score.

The question tests whether a coding assistant remembers details from past \
development sessions — things like library versions, file paths, config \
values, bug fixes, architecture decisions, and coding conventions.

Grading guidelines:
- Be generous: if the generated answer captures the essential fact from the \
gold answer, mark CORRECT even if it includes extra detail or different wording
- IMPORTANT: If the generated answer says it cannot find the information, \
doesn't know, says the evidence is not in the transcripts, or otherwise \
fails to answer the question, mark WRONG — even if the question asks about \
something that was "never implemented" or "never changed"
- For file paths: treat "src/foo.py" and "./src/foo.py" as equivalent
- For version numbers: "2.1" and "v2.1" and "2.1.0" are equivalent
- For code: if the logic is equivalent even with different variable names or \
formatting, mark CORRECT
- For temporal questions: if the answer identifies the correct time period \
(same week/month/session), mark CORRECT even if the exact date differs slightly
- For "why" questions: the answer must capture the core reasoning, not just \
restate the decision

Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, \
then finish with CORRECT or WRONG. Do NOT include both CORRECT and WRONG \
in your response.

Return the label in JSON format with the key "label"."""


# ---------------------------------------------------------------------------
# Utility functions (shared with locomo_eval.py)
# ---------------------------------------------------------------------------

def _is_anthropic_model(model: str) -> bool:
    return model.startswith("claude-")


def _is_gpt5_chat_model(model: str) -> bool:
    return model.startswith("gpt-5")


def _api_call_with_retry(client, messages, max_tokens=100, retries=10, model="gpt-4o-mini"):
    """Call OpenAI or Anthropic API with rate-limit-aware retry."""
    for attempt in range(retries):
        try:
            if _is_anthropic_model(model):
                response = client.messages.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                )
                return response.content[0].text.strip()
            else:
                kwargs = {
                    "model": model,
                    "messages": messages,
                }
                # GPT-5 chat models reject max_tokens and some legacy chat knobs.
                if _is_gpt5_chat_model(model):
                    kwargs["max_completion_tokens"] = max_tokens
                    kwargs["reasoning_effort"] = "minimal"
                else:
                    kwargs["temperature"] = 0.0
                    kwargs["max_tokens"] = max_tokens
                    kwargs["seed"] = 42
                response = client.chat.completions.create(**kwargs)
                return response.choices[0].message.content.strip()
        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "rate" in err_str.lower()
            if attempt < retries - 1:
                if is_rate_limit:
                    # RPD resets daily; wait longer and keep trying
                    wait = min(30 * (attempt + 1), 300)
                    print(f"    [rate limit] waiting {wait}s (attempt {attempt+1}/{retries})")
                else:
                    wait = 2 ** attempt
                time.sleep(wait)
                continue
            raise RuntimeError(f"API call failed after {retries} retries: {e}") from e


def normalize_answer(s: str) -> str:
    """Normalize answer for F1 computation."""
    s = str(s).lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    s = " ".join(s.split())
    return s


def token_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def generate_answer(question: str, context: str, client, model: str = "gpt-4o-mini") -> str:
    """Generate a short answer using the configured LLM."""
    prompt = ANSWER_PROMPT.format(context=context, question=question)
    max_tokens = 150 if _is_gpt5_chat_model(model) else 100
    return _api_call_with_retry(
        client, [{"role": "user", "content": prompt}],
        max_tokens=max_tokens, model=model,
    )


def judge_answer(
    question: str, gold_answer: str, generated_answer: str,
    client, model: str = "gpt-4o-mini",
) -> int:
    """Judge whether the generated answer is correct. Returns 1 or 0."""
    prompt = JUDGE_PROMPT.format(
        question=question,
        gold_answer=str(gold_answer),
        generated_answer=generated_answer,
    )
    text = _api_call_with_retry(
        client, [{"role": "user", "content": prompt}],
        max_tokens=100, model=model,
    )
    try:
        result = json.loads(text)
        label = result.get("label", "").upper()
    except json.JSONDecodeError:
        label = "CORRECT" if "CORRECT" in text.upper() else "WRONG"
    return 1 if label == "CORRECT" else 0


def _is_negative_evidence_question(question: str, gold_answer: str) -> bool:
    """Return True when a question explicitly tests absence/negative evidence."""
    text = f"{question}\n{gold_answer}"
    return bool(_NEGATIVE_EVIDENCE_RE.search(text))


def _apply_recall_gate(
    *,
    j_score: int,
    recall_at_k: float,
    question: str,
    gold_answer: str,
) -> int:
    """Cap judged wins that were produced with zero retrieved evidence.

    Exception: questions that explicitly test negative evidence ("never",
    "not implemented", etc.) are allowed to pass without retrieved support.
    """
    if j_score != 1:
        return j_score
    if recall_at_k > 0.0:
        return j_score
    if _is_negative_evidence_question(question, gold_answer):
        return j_score
    return 0


# ---------------------------------------------------------------------------
# Retrieval recall (evidence-based)
# ---------------------------------------------------------------------------

def _parse_retrieved_turns(retrieved_text: str) -> set[tuple[str, int]]:
    """Extract (session_prefix, turn_index) pairs from formatted retrieval output.

    The lookup() format includes headers like:
        --- [2026-03-10 12:00 session session_001] turn 42 ---

    Returns set of (session_id_prefix, turn_index) tuples.
    """
    turn_pattern = re.compile(
        r"session\s+(\S+)\]\s+turn\s+(\d+)(?:,\s+[^\n-][^\n]*)?\s+---"
    )
    retrieved_turns: set[tuple[str, int]] = set()
    for m in turn_pattern.finditer(retrieved_text):
        sid_prefix = m.group(1)
        turn_idx = int(m.group(2))
        retrieved_turns.add((sid_prefix, turn_idx))
    return retrieved_turns


_ALIGN_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]+")


def _normalize_alignment_token(token: str) -> str:
    token = token.lower().strip(".,:;!?()[]{}\"'`")
    if token.isalpha() and token.endswith("s") and len(token) > 4 and not token.endswith("ss"):
        token = token[:-1]
    return token


def _alignment_tokens(text: str) -> set[str]:
    """Extract lightweight lexical tokens for evidence/chunk alignment."""
    return {
        norm
        for tok in _ALIGN_TOKEN_RE.findall(text)
        if (norm := _normalize_alignment_token(tok)) and len(norm) >= 4
    }


def _alignment_score(reference_text: str, chunk_text: str) -> float:
    """Return a simple overlap score between evidence text and chunk text."""
    ref = _alignment_tokens(reference_text)
    if not ref:
        return 0.0
    chunk = _alignment_tokens(chunk_text)
    if not chunk:
        return 0.0
    return len(ref & chunk) / len(ref)


def _align_evidence_chunk_turn(
    evidence: dict,
    raw_turn_text: str,
    direct_turn_index: int | None,
    session_chunk_texts: dict[int, str],
) -> int | None:
    """Align raw evidence to the best matching parsed chunk in the same session.

    Some benchmark evidence points at raw JSONL lines that are compaction
    summaries or tool-result scaffolding, while retrieval surfaces a later
    assistant chunk that carries the same substantive evidence more clearly.
    Prefer the direct raw-line->chunk map when it remains a good lexical match,
    otherwise promote to a clearly better chunk in the same session.
    """
    if not session_chunk_texts:
        return direct_turn_index

    description = str(evidence.get("description", "")).strip()
    evidence_text = description or raw_turn_text.strip()
    if not evidence_text:
        return direct_turn_index

    direct_score = 0.0
    if direct_turn_index is not None:
        direct_score = _alignment_score(
            evidence_text, session_chunk_texts.get(direct_turn_index, "")
        )

    best_turn = direct_turn_index
    best_score = direct_score
    for turn_index, chunk_text in session_chunk_texts.items():
        score = _alignment_score(evidence_text, chunk_text)
        if score > best_score:
            best_turn = turn_index
            best_score = score

    # Only override the direct mapping when the alternative is clearly better.
    if best_turn != direct_turn_index and best_score >= 0.35 and best_score >= direct_score + 0.15:
        return best_turn
    return direct_turn_index


def compute_retrieval_recall(
    retrieved_text: str,
    evidence: list[dict],
    session_texts: dict[str, list[str]],
    chunk_line_map: dict[str, dict[int, int]] | None = None,
    chunk_text_map: dict[str, dict[int, str]] | None = None,
) -> float:
    """Check how many evidence chunks were retrieved.

    Parses the retrieval output headers to find which (session, turn) pairs
    were returned, then checks if each evidence entry's chunk is among them.

    Args:
        retrieved_text: The formatted context returned by the SUT.
        evidence: List of evidence dicts with session_id, turn_index, etc.
        session_texts: Map from session_id to list of turn texts (indexed by
            turn_index — raw JSONL line numbers). Used as fallback.
        chunk_line_map: Optional map from session_id to {raw_line_num: chunk_turn_index}.
            When provided, maps evidence raw line numbers to chunk turn indices
            for precise header-based matching.

    Returns:
        Fraction of evidence turns found in the retrieved text (0.0 to 1.0).
    """
    if not evidence:
        return 0.0

    # Parse retrieved turn headers
    retrieved_turns = _parse_retrieved_turns(retrieved_text)

    found = 0
    for ev in evidence:
        sid = ev.get("session_id", "")
        tidx = ev.get("turn_index", -1)

        raw_turn_text = ""
        turns = session_texts.get(sid, [])
        if 0 <= tidx < len(turns):
            raw_turn_text = turns[tidx].strip()

        # Tier 1: header-based matching with chunk mapping + evidence alignment
        if chunk_line_map and sid in chunk_line_map:
            chunk_tidx = chunk_line_map[sid].get(tidx)
            if chunk_text_map and sid in chunk_text_map:
                chunk_tidx = _align_evidence_chunk_turn(
                    ev,
                    raw_turn_text,
                    chunk_tidx,
                    chunk_text_map[sid],
                )
            if chunk_tidx is not None:
                # Match against session prefix (session IDs may be truncated
                # to 8 chars in the header)
                for ret_prefix, ret_tidx in retrieved_turns:
                    if sid.startswith(ret_prefix) and ret_tidx == chunk_tidx:
                        found += 1
                        break
                continue

        # Tier 2: direct header matching (if turn_index is already a chunk index)
        for ret_prefix, ret_tidx in retrieved_turns:
            if sid.startswith(ret_prefix) and ret_tidx == tidx:
                found += 1
                break
        else:
            # Tier 3: raw-line text fallback
            if raw_turn_text and raw_turn_text[:80] in retrieved_text:
                found += 1
            elif chunk_text_map and sid in chunk_text_map:
                aligned_turn = _align_evidence_chunk_turn(
                    ev,
                    raw_turn_text,
                    chunk_line_map.get(sid, {}).get(tidx) if chunk_line_map else None,
                    chunk_text_map[sid],
                )
                if aligned_turn is not None:
                    aligned_text = chunk_text_map[sid].get(aligned_turn, "").strip()
                    if aligned_text and aligned_text[:80] in retrieved_text:
                        found += 1
            elif raw_turn_text:
                text = raw_turn_text
                if text and text[:80] in retrieved_text:
                    found += 1

    return found / len(evidence)


def _build_chunk_maps(
    session_dir: Path,
) -> tuple[dict[str, dict[int, int]], dict[str, dict[int, str]]]:
    """Build raw-line and parsed-chunk maps for retrieval-recall alignment.

    For each session, parses the transcript into chunks and builds a
    line-number → byte-offset map. Then for each raw line, finds which
    chunk's byte range contains it.

    Returns:
        Tuple of:
        - session_id -> {raw_line_number: chunk_turn_index}
        - session_id -> {chunk_turn_index: chunk_text}
    """
    from synapt.recall.core import parse_transcript

    line_map_result: dict[str, dict[int, int]] = {}
    chunk_text_result: dict[str, dict[int, str]] = {}
    for sp in sorted(session_dir.glob("*.jsonl")):
        sid = sp.stem
        chunks = parse_transcript(sp)
        if not chunks:
            continue

        chunk_text_result[sid] = {c.turn_index: c.text for c in chunks}

        # Build line → byte offset map
        line_map: dict[int, int] = {}
        offset = 0
        line_num = 0
        with open(sp, "rb") as f:
            for raw_line in f:
                line_map[line_num] = offset
                offset += len(raw_line)
                line_num += 1

        # Single-pass: iterate sorted lines against sorted chunks
        raw_to_chunk: dict[int, int] = {}
        sorted_chunks = sorted(chunks, key=lambda c: c.byte_offset)
        ci = 0
        for ln in range(line_num):
            byte_off = line_map[ln]
            # Advance chunk pointer past chunks that end before this line
            while ci < len(sorted_chunks) - 1:
                c = sorted_chunks[ci]
                if byte_off < c.byte_offset + c.byte_length:
                    break
                ci += 1
            c = sorted_chunks[ci]
            if c.byte_offset <= byte_off < c.byte_offset + c.byte_length:
                raw_to_chunk[ln] = c.turn_index

        line_map_result[sid] = raw_to_chunk

    return line_map_result, chunk_text_result


def load_session_texts(session_dir: Path) -> dict[str, list[str]]:
    """Load all session transcripts and return a map of session_id -> turn texts."""
    texts: dict[str, list[str]] = {}
    for sp in sorted(session_dir.glob("*.jsonl")):
        session_id = sp.stem
        turn_texts = []
        for line in open(sp, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                content = ""
                if isinstance(msg.get("message"), dict):
                    for block in msg["message"].get("content", []):
                        if isinstance(block, dict) and block.get("type") == "text":
                            content += block.get("text", "")
                        elif isinstance(block, dict) and block.get("type") == "tool_result":
                            tr = block.get("content", "")
                            if isinstance(tr, str):
                                content += tr
                            elif isinstance(tr, list):
                                for sub in tr:
                                    if isinstance(sub, dict) and sub.get("type") == "text":
                                        content += sub.get("text", "")
                                    elif isinstance(sub, str):
                                        content += sub
                        elif isinstance(block, str):
                            content += block
                turn_texts.append(content)
            except json.JSONDecodeError:
                turn_texts.append("")
        texts[session_id] = turn_texts
    return texts


# ---------------------------------------------------------------------------
# Project discovery
# ---------------------------------------------------------------------------

def discover_projects(project_filter: str | None = None) -> list[Path]:
    """Find all project directories under data/."""
    projects = []
    data_dir = _get_data_dir()
    if not data_dir.exists():
        return projects
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir():
            continue
        if not (d / "manifest.json").exists():
            continue
        if not (d / "questions.json").exists():
            continue
        if project_filter and d.name != project_filter:
            continue
        projects.append(d)
    return projects


# ---------------------------------------------------------------------------
# Repeat-run helpers
# ---------------------------------------------------------------------------

def _result_key(result: dict) -> str:
    """Stable checkpoint key across repeat runs and projects."""
    run = int(result.get("run", 1) or 1)
    project = result.get("project", "")
    qid = result.get("id", "")
    return f"{run}:{project}:{qid}"


def _build_run_summary(
    *,
    run: int,
    project_ids: list[str],
    max_chunks: int,
    model: str,
    pipeline_mode: str,
    provenance: dict,
    retrieval_only: bool,
    category_scores: dict[int, list],
    category_f1: dict[int, list],
    category_recall: dict[int, list],
    total_retrieve_time: float,
    total_generate_time: float,
    total_judge_time: float,
    total_chunks: int,
    total_knowledge: int,
    question_count: int,
    enrich_model_name: str = "",
) -> dict:
    """Build one pass summary without mutating outer state."""
    summary: dict = {
        "benchmark": "codememo",
        "run": run,
        "projects": project_ids,
        "questions_evaluated": question_count,
        "max_chunks": max_chunks,
        "index_chunks": total_chunks,
        "model": model,
        "enrichment_model": enrich_model_name or "none",
        "pipeline_mode": pipeline_mode,
        "provenance": provenance,
    }
    if total_knowledge:
        summary["knowledge_nodes"] = total_knowledge
    if question_count:
        summary["avg_retrieve_ms"] = round(total_retrieve_time / question_count * 1000, 1)

    for cat in sorted(ALL_CATEGORIES):
        if category_recall[cat]:
            avg = sum(category_recall[cat]) / len(category_recall[cat])
            summary[f"recall@{max_chunks}_{CATEGORY_NAMES[cat]}"] = round(avg * 100, 2)
            summary[f"n_{CATEGORY_NAMES[cat]}"] = len(category_recall[cat])

    if not retrieval_only and category_scores:
        if question_count:
            summary["avg_generate_ms"] = round(total_generate_time / question_count * 1000, 1)
            summary["avg_judge_ms"] = round(total_judge_time / question_count * 1000, 1)

        for cat in sorted(ALL_CATEGORIES):
            if category_scores[cat]:
                j = sum(category_scores[cat]) / len(category_scores[cat]) * 100
                summary[f"j_score_{CATEGORY_NAMES[cat]}"] = round(j, 2)

        all_j = [s for scores in category_scores.values() for s in scores]
        summary["j_score_overall"] = round(sum(all_j) / len(all_j) * 100, 2) if all_j else 0

        for cat in sorted(ALL_CATEGORIES):
            if category_f1[cat]:
                avg_f1 = sum(category_f1[cat]) / len(category_f1[cat]) * 100
                summary[f"f1_{CATEGORY_NAMES[cat]}"] = round(avg_f1, 2)

        all_f1 = [f for scores in category_f1.values() for f in scores]
        summary["f1_overall"] = round(sum(all_f1) / len(all_f1) * 100, 2) if all_f1 else 0

    return summary


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    project_dirs: list[Path],
    sut: SystemUnderTest,
    retrieval_only: bool = False,
    max_chunks: int = 20,
    output_path: Path | None = None,
    model: str = "gpt-4o-mini",
    repeat_runs: int = 1,
    reset_working_memory_between_runs: bool = False,
) -> dict:
    """Run the CodeMemo evaluation across one or more projects.

    Args:
        project_dirs: List of project directories to evaluate.
        sut: SystemUnderTest implementation.
        retrieval_only: Skip answer generation and judging.
        max_chunks: Chunks to retrieve per question.
        output_path: Directory for results JSON.
        model: OpenAI model for answer generation and judging.

    Returns:
        Summary dict with per-category and overall scores.
    """
    client = None
    if not retrieval_only:
        try:
            if _is_anthropic_model(model):
                from anthropic import Anthropic
                client = Anthropic()
            else:
                from openai import OpenAI
                client = OpenAI()
        except Exception as e:
            print(f"API client failed: {e}")
            print("Falling back to retrieval-only mode.")
            retrieval_only = True

    # Provenance + audit
    provenance = _get_provenance()
    pipeline_mode = sut.mode if isinstance(sut, SynaptSUT) else "external"
    audit_ts = _audit_start(pipeline_mode, model, provenance)

    all_results: list[dict] = []

    # Resume from checkpoint if available
    checkpoint_dir = output_path if output_path else Path("evaluation/codememo/results")
    checkpoint_path = checkpoint_dir / "codememo_checkpoint.json"
    completed_ids: set[str] = set()
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            checkpoint_data = json.load(f)
        all_results = checkpoint_data
        for r in all_results:
            completed_ids.add(_result_key(r))
        print(f"  Resumed from checkpoint: {len(completed_ids)} completed question-runs")

    project_infos: list[dict] = []
    total_chunks = 0
    total_knowledge = 0

    for project_dir in project_dirs:
        project_id = project_dir.name
        print(f"\n{'='*60}")
        print(f"Project: {project_id}")
        print(f"{'='*60}")

        # Load manifest
        with open(project_dir / "manifest.json") as f:
            manifest = json.load(f)
        print(f"  {manifest.get('description', '')}")
        print(f"  Tech: {', '.join(manifest.get('tech_stack', []))}")

        # Load questions
        with open(project_dir / "questions.json") as f:
            questions = json.load(f)
        print(f"  Questions: {len(questions)}")

        # Find session files
        session_dir = project_dir / "sessions"
        session_paths = sorted(session_dir.glob("*.jsonl"))
        if not session_paths:
            print(f"  WARNING: No session transcripts found in {session_dir}")
            print(f"  Skipping project (transcripts not yet generated)")
            continue

        # Load session texts for retrieval recall
        session_texts = load_session_texts(session_dir)

        # Build chunk maps for precise retrieval recall
        chunk_line_map, chunk_text_map = _build_chunk_maps(session_dir)

        # Ingest sessions into SUT
        print(f"  Ingesting {len(session_paths)} sessions...")
        t0 = time.time()
        sut.ingest(session_paths)
        ingest_time = time.time() - t0
        stats = sut.stats() if hasattr(sut, "stats") else {}
        chunk_count = stats.get("chunk_count", 0)
        kn_count = stats.get("knowledge_count", 0)
        total_chunks += chunk_count
        total_knowledge += kn_count
        kn_msg = f", {kn_count} knowledge nodes" if kn_count else ""
        print(f"  Indexed: {chunk_count} chunks{kn_msg} in {ingest_time:.1f}s")

        project_infos.append({
            "project_id": project_id,
            "questions": questions,
            "session_texts": session_texts,
            "chunk_line_map": chunk_line_map,
            "chunk_text_map": chunk_text_map,
        })

    # ---------------------------------------------------------------------------
    # Aggregate scores
    # ---------------------------------------------------------------------------
    run_summaries: list[dict] = []
    total_questions_all = 0

    # Resolve enrichment model for provenance
    enrich_model_name = ""
    if isinstance(sut, SynaptSUT):
        enrich_model_name = sut.enrich_model
        if not enrich_model_name and sut.mode == "full-pipeline":
            try:
                from synapt.recall._model_router import DEFAULT_DECODER_MODEL
                enrich_model_name = DEFAULT_DECODER_MODEL
            except ImportError:
                enrich_model_name = "(default — unknown)"

    for run in range(1, max(1, repeat_runs) + 1):
        if run > 1 and reset_working_memory_between_runs and hasattr(sut, "reset_working_memory"):
            sut.reset_working_memory()

        category_scores: dict[int, list] = defaultdict(list)
        category_f1: dict[int, list] = defaultdict(list)
        category_recall: dict[int, list] = defaultdict(list)
        total_retrieve_time = 0.0
        total_generate_time = 0.0
        total_judge_time = 0.0
        total_questions = 0

        print(f"\n{'-'*60}")
        print(f"Run {run}/{max(1, repeat_runs)}")
        print(f"{'-'*60}")

        for info in project_infos:
            project_id = info["project_id"]
            questions = info["questions"]
            session_texts = info["session_texts"]
            chunk_line_map = info["chunk_line_map"]
            chunk_text_map = info["chunk_text_map"]

            for i, qa in enumerate(questions):
                qid = qa["id"]
                question = qa["question"]
                gold = str(qa.get("answer_short", qa["answer"]))
                category = qa["category"]
                evidence = qa.get("evidence", [])

                if category not in ALL_CATEGORIES:
                    continue

                key = f"{run}:{project_id}:{qid}"
                if key in completed_ids:
                    continue

                t0 = time.time()
                context = sut.query(question, max_chunks=max_chunks)
                retrieve_time = time.time() - t0
                total_retrieve_time += retrieve_time

                recall_at_k = compute_retrieval_recall(
                    context,
                    evidence,
                    session_texts,
                    chunk_line_map=chunk_line_map,
                    chunk_text_map=chunk_text_map,
                )
                category_recall[category].append(recall_at_k)

                result_entry = {
                    "run": run,
                    "project": project_id,
                    "id": qid,
                    "question": question,
                    "gold_answer": gold,
                    "category": category,
                    "category_name": CATEGORY_NAMES.get(category, "?"),
                    "context": context,
                    "context_length": len(context),
                    "retrieve_time_ms": round(retrieve_time * 1000, 1),
                    "retrieval_recall": recall_at_k,
                }

                if not retrieval_only and client:
                    t0 = time.time()
                    generated = generate_answer(question, context, client, model=model)
                    generate_time = time.time() - t0
                    total_generate_time += generate_time

                    f1 = token_f1(generated, gold)
                    category_f1[category].append(f1)

                    t0 = time.time()
                    raw_j_score = judge_answer(question, gold, generated, client, model=model)
                    judge_time = time.time() - t0
                    total_judge_time += judge_time

                    j_score = _apply_recall_gate(
                        j_score=raw_j_score,
                        recall_at_k=recall_at_k,
                        question=question,
                        gold_answer=gold,
                    )
                    category_scores[category].append(j_score)
                    result_entry.update({
                        "generated_answer": generated,
                        "j_score": j_score,
                        "raw_j_score": raw_j_score,
                        "f1": round(f1, 4),
                        "generate_time_ms": round(generate_time * 1000, 1),
                        "judge_time_ms": round(judge_time * 1000, 1),
                    })

                all_results.append(result_entry)
                completed_ids.add(key)
                total_questions += 1
                total_questions_all += 1

                if (i + 1) % 10 == 0 or i == len(questions) - 1:
                    elapsed = total_retrieve_time + total_generate_time + total_judge_time
                    if not retrieval_only and category_scores:
                        all_j = [s for scores in category_scores.values() for s in scores]
                        running_j = sum(all_j) / len(all_j) * 100 if all_j else 0
                        print(
                            f"  Run {run} [{i+1}/{len(questions)}] "
                            f"J={running_j:.1f}% ({elapsed:.0f}s elapsed)"
                        )
                    else:
                        print(
                            f"  Run {run} [{i+1}/{len(questions)}] retrieval done "
                            f"({elapsed:.0f}s elapsed)"
                        )

                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    with open(checkpoint_path, "w") as f:
                        json.dump(all_results, f)

        run_summaries.append(_build_run_summary(
            run=run,
            project_ids=[d.name for d in project_dirs],
            max_chunks=max_chunks,
            model=model,
            pipeline_mode=sut.mode if isinstance(sut, SynaptSUT) else "external",
            provenance=provenance,
            retrieval_only=retrieval_only,
            category_scores=category_scores,
            category_f1=category_f1,
            category_recall=category_recall,
            total_retrieve_time=total_retrieve_time,
            total_generate_time=total_generate_time,
            total_judge_time=total_judge_time,
            total_chunks=total_chunks,
            total_knowledge=total_knowledge,
            question_count=total_questions,
            enrich_model_name=enrich_model_name,
        ))

    summary = dict(run_summaries[-1]) if run_summaries else {
        "benchmark": "codememo",
        "projects": [d.name for d in project_dirs],
        "questions_evaluated": 0,
        "max_chunks": max_chunks,
        "model": model,
        "pipeline_mode": sut.mode if isinstance(sut, SynaptSUT) else "external",
        "provenance": provenance,
    }
    summary["repeat_runs"] = max(1, repeat_runs)
    summary["questions_per_run"] = run_summaries[-1]["questions_evaluated"] if run_summaries else 0
    summary["questions_evaluated_total"] = total_questions_all
    if run_summaries:
        summary["run_summaries"] = run_summaries
        if len(run_summaries) > 1:
            deltas: list[dict] = []
            for prev, curr in zip(run_summaries, run_summaries[1:]):
                delta = {"from_run": prev["run"], "to_run": curr["run"]}
                if "j_score_overall" in prev and "j_score_overall" in curr:
                    delta["j_score_overall_delta"] = round(
                        curr["j_score_overall"] - prev["j_score_overall"], 2
                    )
                deltas.append(delta)
            summary["run_deltas"] = deltas
    summary = _add_validity_metadata(summary)

    # Print summary
    print("\n" + "=" * 60)
    print("CODEMEMO BENCHMARK RESULTS")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    if summary.get("validity_notes"):
        print("  benchmark_validity:")
        for note in summary["validity_notes"]:
            print(f"    - {note}")
    print("=" * 60)

    # Save results
    if output_path is None:
        output_path = Path("evaluation/codememo/results")
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "codememo_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(output_path / "codememo_detailed.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}/")

    _audit_finish(audit_ts, summary)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CodeMemo benchmark for coding-assistant memory systems")
    parser.add_argument(
        "--project", type=str, default=None,
        help="Evaluate a single project (directory name under data/)",
    )
    parser.add_argument(
        "--retrieval-only", action="store_true",
        help="Only measure retrieval quality (no API key needed)",
    )
    parser.add_argument(
        "--max-chunks", type=int, default=20,
        help="Number of chunks to retrieve per question (default: 20)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--recalldb", action="store_true",
        help="Use RecallDB + FTS5 + embeddings (fast, no MLX enrichment)",
    )
    parser.add_argument(
        "--full-pipeline", action="store_true",
        help="Use full pipeline: RecallDB + enrich + consolidate + knowledge graph",
    )
    parser.add_argument(
        "--enrich-model", type=str, default="",
        help="MLX model for enrichment/consolidation",
    )
    parser.add_argument(
        "--max-knowledge", type=int, default=None,
        help="Maximum knowledge blocks per query",
    )
    parser.add_argument(
        "--knowledge-boost", type=float, default=None,
        help="Override knowledge node boost factor",
    )
    # Load default judge model from eval_config.json if available
    _eval_config_model = "gpt-4o-mini"
    _eval_config_path = Path(__file__).parent.parent / "eval_config.json"
    if _eval_config_path.exists():
        try:
            _eval_config_model = json.loads(_eval_config_path.read_text()).get(
                "codememo", {}
            ).get("judge_model", _eval_config_model)
        except Exception:
            pass
    parser.add_argument(
        "--model", type=str, default=_eval_config_model,
        help=f"OpenAI model for answer generation and judging (default: {_eval_config_model} from eval_config.json)",
    )
    parser.add_argument(
        "--work-dir", type=str, default="",
        help="Persistent work directory for index/enrichment data. "
             "If set, reused across runs (GPU enrich → CPU eval split).",
    )
    parser.add_argument(
        "--repeat-runs", type=int, default=1,
        help="Run the same question set multiple times against the same ingested DB. "
             "Useful for measuring persistent access-frequency effects.",
    )
    parser.add_argument(
        "--reset-working-memory-between-runs", action="store_true",
        help="Clear only session-scoped working memory between repeat passes while "
             "keeping persistent access-frequency boosts intact.",
    )
    args = parser.parse_args()

    # Discover projects
    project_dirs = discover_projects(args.project)
    if not project_dirs:
        print(f"No projects found in {_get_data_dir()}")
        if args.project:
            print(f"  (filtered for: {args.project})")
        sys.exit(1)

    print(f"Found {len(project_dirs)} project(s): "
          f"{', '.join(d.name for d in project_dirs)}")

    # Set up SUT
    if args.full_pipeline:
        mode = "full-pipeline"
    elif args.recalldb:
        mode = "recalldb"
    else:
        mode = "baseline"

    sut = SynaptSUT(
        mode=mode,
        enrich_model=args.enrich_model,
        max_knowledge=args.max_knowledge,
        knowledge_boost=args.knowledge_boost,
        work_dir=args.work_dir,
    )

    try:
        run_evaluation(
            project_dirs=project_dirs,
            sut=sut,
            retrieval_only=args.retrieval_only,
            max_chunks=args.max_chunks,
            output_path=args.output,
            model=args.model,
            repeat_runs=args.repeat_runs,
            reset_working_memory_between_runs=args.reset_working_memory_between_runs,
        )
    except Exception as e:
        # Audit failure — find the most recent RUNNING entry's timestamp
        _audit_log({
            "type": "eval", "eval_type": "codememo",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "outcome": "FAILED",
            "error": str(e)[:200],
        })
        raise
    finally:
        sut.close()


if __name__ == "__main__":
    main()
