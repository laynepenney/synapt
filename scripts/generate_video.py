#!/usr/bin/env python3
"""Generate a blog hero video from a hero image via Seedance 2.0.

Uses fal.ai's Seedance 2.0 image-to-video model to animate a static hero
image into a 5-second ambient background video.

Usage:
    python scripts/generate_video.py the-boundary-is-the-business
    python scripts/generate_video.py my-post --duration 8
    python scripts/generate_video.py my-post --prompt "custom animation prompt"
    python scripts/generate_video.py my-post --image /path/to/custom-source.png

Requires:
    - FAL_KEY environment variable (or in ~/.zshrc)
    - fal_client (pip install fal-client)
    - requests
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

try:
    import fal_client
except ImportError:
    print("Missing dependency: pip install fal-client")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("Missing dependency: pip install requests")
    sys.exit(1)

BLOG_DIR = Path(__file__).resolve().parent.parent / "docs" / "blog"
IMAGES_DIR = BLOG_DIR / "images"
DEFAULT_PROMPT = (
    "subtle holographic shimmer, mesh lines pulse gently, particles drift, "
    "atmospheric, slow, cinematic"
)
VALID_DURATIONS = ["4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]


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


def find_hero_raw(slug: str) -> Path:
    """Find the raw (pre-watermark) hero image for a given slug."""
    raw = IMAGES_DIR / f"{slug}-hero-raw.png"
    if raw.exists():
        return raw
    hero = IMAGES_DIR / f"{slug}-hero.png"
    if hero.exists():
        print(f"  Warning: using watermarked hero (no raw found at {raw.name})")
        return hero
    print(f"Error: no hero image found for slug '{slug}'")
    print(f"  Looked for: {raw}")
    print(f"  Looked for: {hero}")
    sys.exit(1)


def generate_video(image_path: Path, prompt: str, duration: str) -> str:
    """Upload image, generate video via Seedance 2.0, return video URL."""
    get_fal_key()

    print(f"Uploading {image_path.name} to fal.ai storage...")
    image_url = fal_client.upload_file(str(image_path))
    print(f"  Uploaded: {image_url}")

    print(f"Submitting to Seedance 2.0 (image-to-video, {duration}s)...")
    print(f"  Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")

    result = fal_client.subscribe(
        "fal-ai/seedance-2/image-to-video",
        arguments={
            "image_url": image_url,
            "prompt": prompt,
            "duration": duration,
        },
        with_logs=True,
    )

    video_url = result["video"]["url"]
    print(f"  Generated: {video_url}")
    return video_url


def download_video(url: str, output_path: Path) -> None:
    """Download video from URL to local path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(8192):
            f.write(chunk)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Downloaded: {output_path} ({size_mb:.1f} MB)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a blog hero video from a hero image via Seedance 2.0"
    )
    parser.add_argument("slug", help="Post slug (e.g. 'the-boundary-is-the-business')")
    parser.add_argument(
        "--image", metavar="PATH",
        help="Source image path (default: images/<slug>-hero-raw.png)",
    )
    parser.add_argument(
        "--prompt", default=DEFAULT_PROMPT,
        help="Animation prompt (default: subtle holographic shimmer)",
    )
    parser.add_argument(
        "--duration", default="5", choices=VALID_DURATIONS,
        help="Video duration in seconds (default: 5)",
    )
    parser.add_argument(
        "--output", "-o", metavar="PATH",
        help="Output path (default: images/<slug>.mp4)",
    )
    args = parser.parse_args()

    image_path = Path(args.image) if args.image else find_hero_raw(args.slug)
    output_path = Path(args.output) if args.output else IMAGES_DIR / f"{args.slug}.mp4"

    url = generate_video(image_path, args.prompt, args.duration)
    download_video(url, output_path)

    print(f"\nDone! Video ready at: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
