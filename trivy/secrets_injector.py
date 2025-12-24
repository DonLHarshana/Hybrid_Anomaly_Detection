# trivy/secrets_injector.py

import os, csv, shutil, base64, argparse, random, string
from pathlib import Path


# Fake secret generators
def fake_aws_key():
    return "AKIA" + "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567", k=16))


def fake_aws_secret():
    # keep realistic secret-like output
    return base64.b64encode(os.urandom(30)).decode()[:40]


def fake_postgres_uri():
    user = "user_" + ''.join(random.choices(string.ascii_lowercase, k=5))
    pwd = base64.b16encode(os.urandom(8)).decode().lower()
    return f"postgres://{user}:{pwd}@db.example.com:5432/app"


def fake_api_key():
    return "sk_test_" + base64.b32encode(os.urandom(12)).decode().strip("=")


def fake_jwt():
    return (
        "eyJhbGciOiJIUzI1NiJ9."
        + base64.urlsafe_b64encode(os.urandom(18)).decode().strip("=")
        + "."
        + base64.urlsafe_b64encode(os.urandom(18)).decode().strip("=")
    )


# All available tokens
TOKENS = {
    "{{AWS_ACCESS_KEY_ID}}": fake_aws_key,
    "{{AWS_SECRET_ACCESS_KEY}}": fake_aws_secret,
    "{{POSTGRES_URI}}": fake_postgres_uri,
    "{{GENERIC_API_KEY}}": fake_api_key,
    "{{DUMMY_JWT}}": fake_jwt,
}

# ✅ NEW: profile configs = (allowed token types) + (MAX injections total)
# This guarantees low/medium/high are separated by number of injected secrets.
PROFILE_CONFIG = {
    "clean":  {"allowed": [], "max_injections": 0},
    "low":    {"allowed": ["{{GENERIC_API_KEY}}"], "max_injections": 1},
    "medium": {"allowed": ["{{POSTGRES_URI}}", "{{GENERIC_API_KEY}}"], "max_injections": 2},
    "high":   {"allowed": ["{{AWS_ACCESS_KEY_ID}}", "{{AWS_SECRET_ACCESS_KEY}}",
                           "{{POSTGRES_URI}}", "{{GENERIC_API_KEY}}", "{{DUMMY_JWT}}"],
               "max_injections": 6},
}


def safe_placeholder(token: str) -> str:
    # Non-secret placeholder so files look complete but Trivy won't flag it
    name = token.strip("{}")
    return f"__{name}_PLACEHOLDER__"


def inject(template_dir: Path, out_dir: Path, gt_csv: Path, profile: str = "medium", fill_unused: bool = True):
    """
    Inject secrets into template files based on profile.

    Key behavior:
    - Profile controls allowed token types AND maximum number of injections.
    - Even if a token appears many times in templates, we inject only up to max_injections.
    - Optionally replace any remaining placeholders with non-secret placeholders (fill_unused=True).

    Args:
        template_dir: Source template directory
        out_dir: Output directory for generated payment set
        gt_csv: Path to ground truth CSV file
        profile: Injection profile (clean, low, medium, high)
        fill_unused: Replace leftover tokens with safe placeholders
    """
    profile = (profile or "medium").lower().strip()
    cfg = PROFILE_CONFIG.get(profile, PROFILE_CONFIG["medium"])
    allowed_tokens = cfg["allowed"]
    max_inj = int(cfg["max_injections"])

    # Copy templates
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(template_dir, out_dir)

    gt_rows = []
    injected = 0

    # Process files in deterministic order
    for p in sorted(out_dir.rglob("*")):
        if p.is_dir():
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue

        changed = False

        # 1) Inject only allowed tokens, up to max_inj
        for token in allowed_tokens:
            if injected >= max_inj:
                break
            gen = TOKENS.get(token)
            if not gen:
                continue

            while token in text and injected < max_inj:
                idx = text.index(token)
                val = gen()
                rel = p.relative_to(out_dir)

                secret_type = token.strip("{}").lower()
                gt_rows.append({
                    "secret_type": secret_type,
                    "file_path": str(rel)
                })

                text = text.replace(token, val, 1)
                injected += 1
                changed = True

        # 2) Replace leftover placeholders with safe non-secret placeholders (recommended)
        if fill_unused:
            for token in TOKENS.keys():
                if token in text:
                    text = text.replace(token, safe_placeholder(token))
                    changed = True

        if changed:
            p.write_text(text, encoding="utf-8")

    # Write ground truth
    gt_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(gt_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["secret_type", "file_path"])
        w.writeheader()
        w.writerows(gt_rows)

    print(f"✓ Profile: {profile}")
    print(f"✓ Max injections allowed: {max_inj}")
    print(f"✓ Actually injected: {len(gt_rows)} secrets into {out_dir}")
    print(f"✓ Ground truth written to {gt_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default="payment_set_template")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--profile", default=None, help="Override injection profile (clean/low/medium/high)")
    ap.add_argument("--no-fill-unused", action="store_true", help="Do not replace unused placeholders")
    a = ap.parse_args()

    # Profile priority: CLI --profile > env INJECT_PROFILE > default medium
    profile = (a.profile or os.getenv("INJECT_PROFILE", "medium")).lower().strip()
    fill_unused = not a.no_fill_unused

    print(f"Using injection profile: {profile}")
    inject(Path(a.template), Path(a.out), Path(a.gt), profile=profile, fill_unused=fill_unused)
