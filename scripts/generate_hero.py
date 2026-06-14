#!/usr/bin/env python3
"""Generate a watermarked blog hero image from a text prompt.

Combines fal.ai image generation (nano-banana-2) with the synapt watermark
into a single command.

Usage:
    python scripts/generate_hero.py "search engine brain with glowing shards" sprint-13-recap
    python scripts/generate_hero.py "holographic owl" my-post --no-watermark
    python scripts/generate_hero.py --prompt-from docs/blog/my-post.md my-post

Requires:
    - FAL_KEY environment variable (or in ~/.zshrc)
    - Pillow (for watermark)
    - requests
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("Missing dependency: pip install requests")
    sys.exit(1)

BLOG_DIR = Path(__file__).resolve().parent.parent / "docs" / "blog"
IMAGES_DIR = BLOG_DIR / "images"
FAL_ENDPOINT = "https://queue.fal.run/fal-ai/nano-banana-2"
DEFAULT_STYLE = (
    "wireframe holographic owl as focal subject on dark background, "
    "teal and purple neon, circuit board aesthetic, digital art"
)
POLL_INTERVAL = 3
MAX_POLLS = 40  # 2 minutes max


def get_fal_key() -> str:
    """Resolve FAL_KEY.

    Env value handling (printenv stays non-secret):
      - `$(op read ...)` literal — bash eval expands the command substitution into the secret
      - direct value — passes through unchanged

    Fallback if env empty: `op read op://<vault>/FAL_KEY/credential` (OP_ACCOUNT, OP_VAULT).
    """
    raw = (os.environ.get("FAL_KEY") or "").strip()
    if raw:
        result = subprocess.run(
            ["bash", "-c", 'eval echo "$FAL_KEY"'],
            env={**os.environ, "FAL_KEY": raw},
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        print(f"Error: FAL_KEY eval failed: {result.stderr.strip() or 'empty result'}", file=sys.stderr)
        sys.exit(1)
    cmd = ["op", "read"]
    if op_account := os.environ.get("OP_ACCOUNT"):
        cmd.append(f"--account={op_account}")
    vault = os.environ.get("OP_VAULT", "Private")
    cmd.append(f"op://{vault}/FAL_KEY/credential")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except FileNotFoundError:
        print("Error: FAL_KEY not in env and `op` CLI not installed.", file=sys.stderr)
        sys.exit(1)
    if result.returncode != 0:
        print(f"Error: `op read op://{vault}/FAL_KEY/credential` failed: {result.stderr.strip()}", file=sys.stderr)
        print("Ensure 1Password is unlocked and CLI integration is enabled.", file=sys.stderr)
        if not os.environ.get("OP_ACCOUNT"):
            print("For multi-account setups, set OP_ACCOUNT (e.g. export OP_ACCOUNT=mysub.1password.com).", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def extract_prompt_from_md(md_path: Path) -> str:
    """Extract a prompt hint from markdown frontmatter 'hero_prompt' field."""
    if not md_path.exists():
        print(f"Error: {md_path} not found")
        sys.exit(1)
    text = md_path.read_text()
    m = re.search(r"^hero_prompt:\s*(.+)$", text, re.MULTILINE)
    if m:
        return m.group(1).strip().strip("\"'")
    # Fall back to title
    m = re.search(r"^title:\s*(.+)$", text, re.MULTILINE)
    if m:
        title = m.group(1).strip().strip("\"'")
        return f"{title}, {DEFAULT_STYLE}"
    print(f"Error: no hero_prompt or title found in {md_path}")
    sys.exit(1)


def generate_image(prompt: str, fal_key: str) -> str:
    """Submit image generation request and poll for result. Returns image URL."""
    headers = {
        "Authorization": f"Key {fal_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "prompt": prompt,
        "image_size": "landscape_16_9",
        "num_images": 1,
    }

    print(f"Submitting to fal.ai...")
    print(f"  Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")

    resp = requests.post(FAL_ENDPOINT, json=payload, headers=headers)
    resp.raise_for_status()
    request_id = resp.json()["request_id"]
    print(f"  Request ID: {request_id}")

    status_url = f"{FAL_ENDPOINT}/requests/{request_id}"
    status_check_url = f"{status_url}/status"

    for i in range(MAX_POLLS):
        time.sleep(POLL_INTERVAL)
        status_resp = requests.get(status_check_url, headers=headers)
        status_resp.raise_for_status()
        status = status_resp.json().get("status", "")
        if status == "COMPLETED":
            break
        print(f"  Polling... ({status})")
    else:
        print("Error: image generation timed out after 2 minutes")
        sys.exit(1)

    result_resp = requests.get(status_url, headers=headers)
    result_resp.raise_for_status()
    images = result_resp.json().get("images", [])
    if not images:
        print("Error: no images returned")
        sys.exit(1)

    url = images[0]["url"]
    print(f"  Generated: {url}")
    return url


def download_image(url: str, output_path: Path) -> None:
    """Download image from URL to local path."""
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(8192):
            f.write(chunk)
    size_kb = output_path.stat().st_size // 1024
    print(f"  Downloaded: {output_path} ({size_kb} KB)")


def watermark_image(raw_path: Path, output_path: Path) -> None:
    """Apply synapt watermark using the watermark script."""
    watermark_script = Path(__file__).parent / "watermark.py"
    if not watermark_script.exists():
        print(f"Warning: {watermark_script} not found, skipping watermark")
        import shutil
        shutil.copy2(raw_path, output_path)
        return

    result = subprocess.run(
        [sys.executable, str(watermark_script), str(raw_path), "--output", str(output_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Watermark failed: {result.stderr}")
        sys.exit(1)
    print(f"  {result.stdout.strip()}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a watermarked blog hero image from a text prompt"
    )
    parser.add_argument("prompt", nargs="?", help="Image generation prompt")
    parser.add_argument("slug", help="Post slug (e.g. 'sprint-13-recap')")
    parser.add_argument(
        "--prompt-from", metavar="MD_FILE",
        help="Extract prompt from markdown frontmatter hero_prompt field",
    )
    parser.add_argument(
        "--style", default=DEFAULT_STYLE,
        help="Style suffix appended to prompt (default: wireframe holographic)",
    )
    parser.add_argument("--no-watermark", action="store_true", help="Skip watermark step")
    parser.add_argument("--no-style", action="store_true", help="Don't append default style suffix")
    args = parser.parse_args()

    if args.prompt_from:
        prompt = extract_prompt_from_md(Path(args.prompt_from))
    elif args.prompt:
        prompt = args.prompt
        if not args.no_style:
            prompt = f"{prompt}, {args.style}"
    else:
        parser.error("Either a prompt or --prompt-from is required")
        return 1

    fal_key = get_fal_key()

    raw_path = IMAGES_DIR / f"{args.slug}-hero-raw.png"
    final_path = IMAGES_DIR / f"{args.slug}-hero.png"

    # Generate
    url = generate_image(prompt, fal_key)

    # Download
    download_image(url, raw_path)

    # Watermark
    if args.no_watermark:
        import shutil
        shutil.copy2(raw_path, final_path)
        print(f"  Copied (no watermark): {final_path}")
    else:
        watermark_image(raw_path, final_path)

    print(f"\nDone! Hero image ready at: {final_path}")
    print(f"Next: python scripts/build_blog.py --force")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
