#!/usr/bin/env python3
"""Generate a blog hero image, watermark it, and create an OG card.

One command, three outputs:
    python scripts/hero.py "sprint-3-recap" "Four owls on circuit boards, sprint dashboard"

Outputs:
    docs/blog/images/{slug}-hero.png          (watermarked hero)
    docs/blog/images/og/{slug}-hero-og.png    (OG social card)

Requires: FAL_KEY in environment or ~/.zshrc, Pillow.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Style guide — append to user prompts for visual consistency across blog posts.
# All hero images should match this aesthetic.
STYLE_SUFFIX = (
    ", wireframe holographic owl as focal subject, "
    "glowing cyan and purple neon lines, "
    "dark ethereal cave background with floating particles, "
    "digital circuitry details, synapt brand aesthetic"
)

# Past prompts that produced good results (verified from recall transcripts):
#
# sprint-3-recap (2026-04-04):
#   "Four holographic wireframe owls perched on glowing circuit boards, each owl a
#    different color (purple, teal, blue, gold), connected by flowing data streams
#    between them, sprint velocity dashboard with glowing metrics in the background,
#    digital teamwork and coordination theme"
#
# building-collaboration (2026-03-20):
#   "Two owls building a bridge between glowing data towers, digital collaboration,
#    dark ethereal background with flowing data streams connecting them"
#
# interview-with-claude (2026-04-02):
#   "A wireframe holographic owl sitting across from a glowing holographic microphone
#    on an ancient stone desk, interview setting, teal and purple neon glow, dark
#    atmospheric background with starry sky, digital wireframe aesthetic,
#    bioluminescent particles"
#
# agent-madness (2026-03-28):
#   "A glowing purple owl with neural network wings versus a dark iron gauntlet with
#    seven different colored energies, epic showdown in a cosmic arena, tournament
#    bracket floating above, dark ethereal atmosphere, no text, no words, no letters"
#
# Key learnings:
# - Add "no text, no words, no letters" when you don't want baked-in text
# - Owls are the synapt mascot — ALWAYS include (enforced via STYLE_SUFFIX)
# - STYLE_SUFFIX auto-appends "wireframe holographic owl" + negative prompts
#   for humans/robots so agents can't accidentally generate non-owl heroes
# - Purple/teal/cyan are brand colors
# - Dark backgrounds with particles/bioluminescence = consistent look

BLOG_IMAGES = Path(__file__).resolve().parent.parent / "docs" / "blog" / "images"
OG_DIR = BLOG_IMAGES / "og"
LOGO_PATH = Path(__file__).resolve().parent.parent / "docs" / "assets" / "logo.png"
FALLBACK_LOGO = Path(__file__).resolve().parent.parent / "assets" / "logo.png"


def _get_fal_key() -> str:
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


def _find_logo() -> Path:
    for p in [Path("/tmp/icon-nobg.png"), LOGO_PATH, FALLBACK_LOGO]:
        if p.exists():
            return p
    raise FileNotFoundError("No logo found")


def generate_hero(prompt: str, fal_key: str) -> str:
    """Submit prompt to fal.ai nano-banana-2, return image URL."""
    print(f"Generating hero image...")
    data = json.dumps({
        "prompt": prompt,
        "image_size": "landscape_16_9",
        "num_images": 1,
    }).encode()

    req = urllib.request.Request(
        "https://queue.fal.run/fal-ai/nano-banana-2",
        data=data,
        headers={
            "Authorization": f"Key {fal_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req) as resp:
        result = json.load(resp)

    request_id = result.get("request_id")
    if not request_id:
        # Synchronous response
        return result["images"][0]["url"]

    # Poll for result
    print(f"  Queued: {request_id}")
    for _ in range(30):
        time.sleep(2)
        poll_req = urllib.request.Request(
            f"https://queue.fal.run/fal-ai/nano-banana-2/requests/{request_id}",
            headers={"Authorization": f"Key {fal_key}"},
        )
        with urllib.request.urlopen(poll_req) as resp:
            poll_result = json.load(resp)
        if "images" in poll_result:
            return poll_result["images"][0]["url"]
        status = poll_result.get("status", "unknown")
        print(f"  Status: {status}")

    raise TimeoutError("Image generation timed out")


def apply_watermark(image_path: Path) -> None:
    """Apply synapt watermark to hero image in-place."""
    hero = Image.open(image_path).convert("RGBA")
    icon = Image.open(_find_logo()).convert("RGBA")
    icon_cropped = icon.crop(icon.getbbox())

    icon_h = max(36, int(hero.height * 0.052))
    aspect = icon_cropped.width / icon_cropped.height
    icon_w = int(icon_h * aspect)
    icon_small = icon_cropped.resize((icon_w, icon_h), Image.LANCZOS)

    font_size = max(26, int(hero.height * 0.039))
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except OSError:
        font = ImageFont.load_default()

    text = "synapt"
    bbox = font.getbbox(text)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    padding_x, padding_y = 10, 8
    pill_w = icon_w + text_w + padding_x * 3 + 4
    pill_h = icon_h + padding_y * 2

    pill = Image.new("RGBA", (pill_w, pill_h), (15, 15, 25, 120))
    draw = ImageDraw.Draw(pill)

    icon_y = padding_y
    pill.paste(icon_small, (padding_x, icon_y), icon_small)

    text_y_offset = bbox[1]
    icon_center_y = icon_y + icon_h // 2
    text_render_y = icon_center_y - text_h // 2 - text_y_offset
    draw.text(
        (icon_w + padding_x * 2 + 4, text_render_y),
        text, fill=(160, 120, 220, 200), font=font,
    )

    hero.paste(pill, (14, 14), pill)
    hero.convert("RGB").save(image_path, quality=95)
    print(f"  Watermarked: {image_path}")


def create_og_card(hero_path: Path, slug: str, title: str, subtitle: str = "") -> Path:
    """Create 1200x630 OG social card from hero image."""
    hero = Image.open(hero_path).convert("RGBA")
    og_w, og_h = 1200, 630
    hero_resized = hero.resize((og_w, og_h), Image.LANCZOS)

    # Dark overlay
    overlay = Image.new("RGBA", (og_w, og_h), (0, 0, 0, 120))
    hero_resized = Image.alpha_composite(hero_resized, overlay)

    draw = ImageDraw.Draw(hero_resized)

    try:
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 42)
        font_sub = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
        font_site = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except OSError:
        font_title = font_sub = font_site = ImageFont.load_default()

    # Title centered
    bbox = draw.textbbox((0, 0), title, font=font_title)
    tw = bbox[2] - bbox[0]
    draw.text(((og_w - tw) / 2, og_h / 2 - 50), title,
              fill=(255, 255, 255, 240), font=font_title)

    # Subtitle
    if subtitle:
        bbox2 = draw.textbbox((0, 0), subtitle, font=font_sub)
        sw = bbox2[2] - bbox2[0]
        draw.text(((og_w - sw) / 2, og_h / 2 + 10), subtitle,
                  fill=(180, 160, 220, 200), font=font_sub)

    # Logo watermark top-left
    icon = Image.open(_find_logo()).convert("RGBA")
    icon_cropped = icon.crop(icon.getbbox())
    icon_h = 28
    aspect = icon_cropped.width / icon_cropped.height
    icon_w = int(icon_h * aspect)
    icon_small = icon_cropped.resize((icon_w, icon_h), Image.LANCZOS)
    hero_resized.paste(icon_small, (20, 18), icon_small)
    draw.text((20 + icon_w + 8, 20), "synapt",
              fill=(160, 120, 220, 200), font=font_site)

    # synapt.dev bottom-right
    draw.text((og_w - 110, og_h - 35), "synapt.dev",
              fill=(160, 120, 220, 180), font=font_site)

    OG_DIR.mkdir(parents=True, exist_ok=True)
    og_path = OG_DIR / f"{slug}-hero-og.png"
    hero_resized.convert("RGB").save(og_path, quality=95)
    print(f"  OG card: {og_path}")
    return og_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate hero image + watermark + OG card in one command",
    )
    parser.add_argument("slug", help="Blog post slug (e.g. sprint-3-recap)")
    parser.add_argument("prompt", help="Image generation prompt (style suffix auto-appended)")
    parser.add_argument("--title", "-t", help="Title for OG card (default: from slug)")
    parser.add_argument("--subtitle", "-s", default="", help="Subtitle for OG card")
    parser.add_argument("--no-style", action="store_true",
                        help="Don't append the brand style suffix to prompt")
    parser.add_argument("--skip-generate", action="store_true",
                        help="Skip image generation, use existing hero")
    args = parser.parse_args()

    hero_path = BLOG_IMAGES / f"{args.slug}-hero.png"
    title = args.title or args.slug.replace("-", " ").title()

    if not args.skip_generate:
        fal_key = _get_fal_key()
        full_prompt = args.prompt if args.no_style else args.prompt + STYLE_SUFFIX
        print(f"  Prompt: {full_prompt[:100]}...")
        url = generate_hero(full_prompt, fal_key)
        print(f"  Downloading: {url}")
        urllib.request.urlretrieve(url, str(hero_path))
        print(f"  Saved: {hero_path}")

    if not hero_path.exists():
        print(f"Error: {hero_path} not found")
        sys.exit(1)

    # Save unwatermarked copy for OG card generation
    raw_path = Path(f"/tmp/{args.slug}-hero-raw.png")
    Image.open(hero_path).save(raw_path)

    apply_watermark(hero_path)
    create_og_card(raw_path, args.slug, title, args.subtitle)

    print(f"\nDone! Files:")
    print(f"  Hero:  {hero_path}")
    print(f"  OG:    {OG_DIR / f'{args.slug}-hero-og.png'}")


if __name__ == "__main__":
    main()
