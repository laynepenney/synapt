"""Run CodeMemo and LOCOMO ablations on Modal — fire and forget.

Usage:
    # Run all ablations
    modal run scripts/modal_eval.py

    # Run a single config
    modal run scripts/modal_eval.py --config cm_full

    # Run all LOCOMO configs
    modal run scripts/modal_eval.py --config lm_

    # Run full-pipeline with GPU enrichment
    modal run scripts/modal_eval.py --config full_pipeline

    # Check results
    modal volume get synapt-eval-vol results/ablation_results.json /tmp/results.json
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
# Path to a locally-checked-out premium extension package, if present (private,
# not part of this public repo). Override via SYNAPT_PREMIUM_LOCAL_DIR if your
# checkout uses a different directory name.
_PREMIUM_LOCAL_DIR = os.environ.get("SYNAPT_PREMIUM_LOCAL_DIR", "premium")
PRIVATE_REPO = REPO_ROOT.parent / _PREMIUM_LOCAL_DIR
LOCOMO_DATASET = PRIVATE_REPO / "evaluation" / "dataset" / "locomo10.json"

app = modal.App("synapt-eval")

volume = modal.Volume.from_name("synapt-eval-vol", create_if_missing=True)

# CPU image — search + judge via OpenAI API
image = (
    modal.Image.debian_slim(python_version="3.13")
    .add_local_dir(
        REPO_ROOT,
        remote_path="/root/synapt",
        copy=True,
        ignore=[".git", "__pycache__", "*.egg-info", ".claude/worktrees"],
    )
    .pip_install(
        "openai>=1.0",
        "huggingface_hub",
    )
    .run_commands("pip install -q /root/synapt")
)

# GPU image — vLLM + Ministral-8B for enrichment
gpu_image = (
    modal.Image.debian_slim(python_version="3.13")
    .add_local_dir(
        REPO_ROOT,
        remote_path="/root/synapt",
        copy=True,
        ignore=[".git", "__pycache__", "*.egg-info", ".claude/worktrees"],
    )
    .pip_install(
        "openai>=1.0",
        "huggingface_hub",
        "vllm",
    )
    .run_commands("pip install -q /root/synapt")
)

# Bundle LOCOMO dataset if available (lives in private repo)
if LOCOMO_DATASET.exists():
    image = image.add_local_file(
        LOCOMO_DATASET,
        remote_path="/root/synapt/evaluation/dataset/locomo10.json",
        copy=True,
    )
    gpu_image = gpu_image.add_local_file(
        LOCOMO_DATASET,
        remote_path="/root/synapt/evaluation/dataset/locomo10.json",
        copy=True,
    )

RESULTS_DIR = "/results"
ENRICH_MODEL = "mistralai/Ministral-8B-Instruct-2410"

# ---------------------------------------------------------------------------
# Environment configs (subtractive ablations)
# ---------------------------------------------------------------------------

FULL = {
    "SYNAPT_SUBCHUNK_MIN_TEXT": "1200",
    "SYNAPT_CROSS_LINK_MAX_EXPAND": "3",
    "SYNAPT_MAX_PER_SESSION": "4",
    "SYNAPT_MAX_SUBCHUNKS_PER_SESSION": "2",
}


def minus(feature: str) -> dict:
    env = {**FULL}
    env[f"SYNAPT_DISABLE_{feature.upper()}"] = "1"
    return env


# (description, env_overrides, extra_args, benchmark, gpu)
CONFIGS = {
    # CodeMemo ablations
    "cm_full": ("Full (all features ON)", FULL, [], "codememo", False),
    "cm_-reranker": ("- reranker", minus("reranker"), [], "codememo", False),
    "cm_-intent": ("- intent classification", minus("intent"), [], "codememo", False),
    "cm_-dedup": ("- deduplication", minus("dedup"), [], "codememo", False),
    "cm_-boosts": ("- working memory boosts", minus("boosts"), [], "codememo", False),
    "cm_-clusters": ("- cluster summaries", minus("clusters"), [], "codememo", False),
    "cm_-content_profile": ("- content profile", minus("content_profile"), [], "codememo", False),
    "cm_-subchunks": ("- sub-chunk splitting", {**FULL, "SYNAPT_SUBCHUNK_MIN_TEXT": "999999999"}, [], "codememo", False),
    "cm_-links": ("- cross-session links", {**FULL, "SYNAPT_CROSS_LINK_MAX_EXPAND": "0"}, [], "codememo", False),
    "cm_full_pipeline": ("Full pipeline (+ enrich)", {**FULL, "SYNAPT_SUMMARY_BACKEND": "vllm"}, ["--full-pipeline", "--enrich-model", ENRICH_MODEL], "codememo", True),
    # LOCOMO ablations
    "lm_full": ("Full (all features ON)", FULL, [], "locomo", False),
    "lm_-reranker": ("- reranker", minus("reranker"), [], "locomo", False),
    "lm_-intent": ("- intent classification", minus("intent"), [], "locomo", False),
    "lm_-dedup": ("- deduplication", minus("dedup"), [], "locomo", False),
    "lm_-boosts": ("- working memory boosts", minus("boosts"), [], "locomo", False),
    "lm_-clusters": ("- cluster summaries", minus("clusters"), [], "locomo", False),
    "lm_-content_profile": ("- content profile", minus("content_profile"), [], "locomo", False),
    "lm_-subchunks": ("- sub-chunk splitting", {**FULL, "SYNAPT_SUBCHUNK_MIN_TEXT": "999999999"}, [], "locomo", False),
    "lm_-links": ("- cross-session links", {**FULL, "SYNAPT_CROSS_LINK_MAX_EXPAND": "0"}, [], "locomo", False),
    "lm_full_pipeline": ("Full pipeline (+ enrich)", {**FULL, "SYNAPT_SUMMARY_BACKEND": "vllm"}, ["--full-pipeline", "--enrich-model", ENRICH_MODEL], "locomo", True),
}


# ---------------------------------------------------------------------------
# GPU function — enrichment only (Ministral-8B via vLLM on A10G)
#
# Runs --retrieval-only --full-pipeline to ingest + enrich + build index
# without any OpenAI API calls. Saves work dir to volume so the CPU step
# can reuse the enriched index.
# ---------------------------------------------------------------------------

@app.function(
    image=gpu_image,
    secrets=[modal.Secret.from_name("openai-secret")],
    volumes={RESULTS_DIR: volume},
    timeout=3600 * 2,
    gpu="A10G",
    memory=32768,
)
def run_enrich(name: str, benchmark: str, env_overrides: dict,
               enrich_model: str, extra_args: list[str] | None = None) -> None:
    """Run enrichment on GPU, persist work dir to volume for CPU eval step."""
    import subprocess
    import sys

    os.environ.update(env_overrides)

    output_dir = Path(RESULTS_DIR) / name
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "work"

    # Check if enrichment already completed
    if (work_dir / "index").exists():
        print(f"SKIP enrichment for {name}: cached at {work_dir}")
        volume.commit()
        return

    volume.reload()

    print(f"=== ENRICH {name} ({benchmark}) on GPU ===")
    print(f"Model: {enrich_model}")

    if benchmark == "codememo":
        cmd = [
            sys.executable, "-m", "evaluation.codememo.eval",
            "--full-pipeline", "--retrieval-only",
            "--enrich-model", enrich_model,
            "--work-dir", str(work_dir),
            "--output", str(output_dir),
        ]
    else:
        cmd = [
            sys.executable, "-m", "evaluation.locomo_eval",
            "--full-pipeline", "--retrieval-only",
            "--enrich-model", enrich_model,
            "--output", str(output_dir),
        ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, cwd="/root/synapt")
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    stderr = proc.stderr.read()
    if proc.returncode != 0:
        print(f"STDERR: {stderr[-2000:]}")

    volume.commit()
    print(f"Enrichment saved to {work_dir}")


# ---------------------------------------------------------------------------
# CPU function — eval (retrieve + generate + judge via OpenAI API)
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("openai-secret")],
    volumes={RESULTS_DIR: volume},
    timeout=3600 * 8,  # 8 hours — rate limits make it slow
    cpu=1,
    memory=2048,
)
def run_ablation(
    name: str, desc: str, env_overrides: dict,
    extra_args: list[str] | None = None, benchmark: str = "codememo",
    use_work_dir: bool = False,
) -> dict:
    """Run a single ablation config (CodeMemo or LOCOMO)."""
    import subprocess
    import sys

    os.environ.update(env_overrides)

    output_dir = Path(RESULTS_DIR) / name
    output_dir.mkdir(parents=True, exist_ok=True)

    if benchmark == "locomo":
        module = "evaluation.locomo_eval"
        summary_file = "locomo_summary.json"
        checkpoint_file = "locomo_checkpoint.json"
        score_key = "j_score_overall"
    else:
        module = "evaluation.codememo.eval"
        summary_file = "codememo_summary.json"
        checkpoint_file = "codememo_checkpoint.json"
        score_key = "j_score_overall"

    summary_path = output_dir / summary_file
    if summary_path.exists():
        result = json.loads(summary_path.read_text())
        if result.get("questions_evaluated", 0) > 0:
            print(f"SKIP {name}: already complete, J={result.get(score_key, '?')}%")
            volume.commit()
            return result

    volume.reload()

    print(f"=== {name}: {desc} ({benchmark}) ===")
    print(f"Env: {json.dumps(env_overrides, indent=2)}")

    cmd = [sys.executable, "-m", module, "--recalldb", "--output", str(output_dir)]
    if benchmark == "codememo":
        cmd += ["--model", "gpt-5-mini"]
    # Point at the persistent work dir from GPU enrichment step
    if use_work_dir and benchmark == "codememo":
        work_dir = output_dir / "work"
        cmd += ["--work-dir", str(work_dir)]
    cmd += extra_args or []

    checkpoint_path = output_dir / checkpoint_file
    last_size = checkpoint_path.stat().st_size if checkpoint_path.exists() else 0

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, cwd="/root/synapt")
    for line in proc.stdout:
        print(line, end="", flush=True)
        if checkpoint_path.exists():
            cur_size = checkpoint_path.stat().st_size
            if cur_size != last_size:
                last_size = cur_size
                volume.commit()
    proc.wait()
    stderr = proc.stderr.read()
    if proc.returncode != 0:
        print(f"STDERR: {stderr[-1000:]}")

    volume.commit()
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    return {"error": f"No summary produced, exit={proc.returncode}"}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("openai-secret")],
    volumes={RESULTS_DIR: volume},
    timeout=3600 * 24,
    cpu=1,
    memory=512,
)
def run_all(config_filter: str | None = None, code_ref: str = ""):
    """Run ablation configs serially, dispatching to CPU or GPU as needed."""
    t_start = time.time()
    if code_ref:
        os.environ["SYNAPT_CODE_REF"] = code_ref

    configs_to_run = CONFIGS
    if config_filter:
        filters = [f.strip() for f in config_filter.split(",")]
        configs_to_run = {
            k: v for k, v in CONFIGS.items()
            if any(f in k for f in filters)
        }

    results = {}
    for name, (desc, env, extra, bench, needs_gpu) in configs_to_run.items():
        print(f"\n{'='*60}")
        print(f"Launching: {name} — {desc} {'[GPU→CPU]' if needs_gpu else '[CPU]'}")
        print(f"{'='*60}")

        # Inject code_ref into env overrides for provenance
        run_env = {**env}
        if code_ref:
            run_env["SYNAPT_CODE_REF"] = code_ref

        if needs_gpu:
            # Step 1: GPU enrichment (minutes, no OpenAI calls)
            print(f"  Step 1: GPU enrichment ({ENRICH_MODEL})")
            run_enrich.remote(name, bench, run_env, ENRICH_MODEL, extra)
            print(f"  Step 1: done — enrichment cached on volume")
            # Step 2: CPU eval reusing enriched work dir (hours, OpenAI calls only)
            result = run_ablation.remote(name, desc, run_env, extra, bench, use_work_dir=True)
        else:
            result = run_ablation.remote(name, desc, run_env, extra, bench)
        results[name] = result

        j = result.get("j_score_overall", "?")
        n = result.get("questions_evaluated", "?")
        print(f">> {name}: J={j}% (n={n})")

    # Save combined results
    elapsed = time.time() - t_start
    combined = {
        "elapsed_min": round(elapsed / 60, 1),
        "configs": {},
    }
    for name, (desc, *_) in configs_to_run.items():
        if name in results and "error" not in results[name]:
            combined["configs"][name] = {
                "description": desc,
                **results[name],
            }

    combined_path = Path(RESULTS_DIR) / "ablation_results.json"
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    volume.commit()

    print(f"\n\n{'='*70}")
    print(f"ABLATION RESULTS ({elapsed/60:.0f}min)")
    print(f"{'='*70}")
    print(f"{'Config':<25s} {'J-Score':>8s}  Description")
    print("-" * 70)
    for name, (desc, *_) in configs_to_run.items():
        if name in results and "j_score_overall" in results[name]:
            print(f"{name:<25s} {results[name]['j_score_overall']:>8}  {desc}")
        else:
            print(f"{name:<25s} {'FAILED':>8}  {desc}")

    print(f"\nSaved to {combined_path}")
    return combined


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(config: str = "", ref: str = ""):
    """Run ablations on Modal.

    Args:
        config: Optional filter — run only configs matching this string.
                Empty = run all.
        ref: Git commit SHA for provenance. Auto-detected if not set.
    """
    # Auto-detect commit SHA from local repo
    if not ref:
        import subprocess
        try:
            ref = subprocess.run(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
            dirty = subprocess.run(
                ["git", "-C", str(REPO_ROOT), "diff", "--quiet"],
                capture_output=True, timeout=5,
            ).returncode
            if dirty:
                ref += "-dirty"
        except Exception:
            ref = "unknown"

    print(f"Code ref: {ref}")

    if config:
        print(f"Running configs matching: {config}")
        result = run_all.remote(config_filter=config, code_ref=ref)
    else:
        print("Running all ablation configs")
        result = run_all.remote(code_ref=ref)

    print(f"\nDone! Results on Modal volume 'synapt-eval-vol'")
    print(f"  modal volume get synapt-eval-vol results/ablation_results.json ./results.json")
