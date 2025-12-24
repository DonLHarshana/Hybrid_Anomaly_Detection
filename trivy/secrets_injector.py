# trivy/secrets_injector.py

import os, csv, shutil, base64, argparse, random, string, io
from pathlib import Path


# Fake secrets generators for non functions 
def fake_aws_key(): 
    return "AKIA" + "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567", k=16))


def fake_aws_secret(): 
    return base64.b64encode(os.urandom(30)).decode()[:40]


def fake_postgres_uri():
    user = "user_" + ''.join(random.choices(string.ascii_lowercase, k=5))
    pwd = base64.b16encode(os.urandom(8)).decode().lower()
    return f"postgres://{user}:{pwd}@db.example.com:5432/app"


def fake_api_key(): 
    return "sk_test_" + base64.b32encode(os.urandom(12)).decode().strip("=")


def fake_jwt():
    return "eyJhbGciOiJIUzI1NiJ9." + base64.urlsafe_b64encode(os.urandom(18)).decode().strip("=") + "." + base64.urlsafe_b64encode(os.urandom(18)).decode().strip("=")


# All available tokens
TOKENS = {
    "{{AWS_ACCESS_KEY_ID}}": fake_aws_key,
    "{{AWS_SECRET_ACCESS_KEY}}": fake_aws_secret,
    "{{POSTGRES_URI}}": fake_postgres_uri,
    "{{GENERIC_API_KEY}}": fake_api_key,
    "{{DUMMY_JWT}}": fake_jwt,
}


# Profile-based injection control
PROFILE_TOKENS = {
    "clean": [],  # No tokens = 0 secrets
    "low": [
        "{{GENERIC_API_KEY}}",  # 1 secret
    ],
    "medium": [
        "{{POSTGRES_URI}}",     # 2 secrets
        "{{GENERIC_API_KEY}}",
    ],
    "high": [
        "{{AWS_ACCESS_KEY_ID}}",      # 6 secrets
        "{{AWS_SECRET_ACCESS_KEY}}",
        "{{POSTGRES_URI}}",
        "{{GENERIC_API_KEY}}",
        "{{DUMMY_JWT}}",
        "{{AWS_ACCESS_KEY_ID}}",  # Repeat for more detections
    ]
}


def inject(template_dir: Path, out_dir: Path, gt_csv: Path, profile: str = "medium"):
    """
    Inject secrets into template files based on profile.
    
    Args:
        template_dir: Source template directory
        out_dir: Output directory for generated payment set
        gt_csv: Path to ground truth CSV file
        profile: Injection profile (clean, low, medium, high)
    """
    # Make a copy of all templates 
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(template_dir, out_dir)
    
    gt_rows = []
    
    # Get allowed tokens for this profile
    allowed_tokens = PROFILE_TOKENS.get(profile, list(TOKENS.keys()))
    
    # Inject every file placeholders with fake secrets 
    for p in out_dir.rglob("*"):
        if p.is_dir():
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        
        changed = False
        for token, gen in TOKENS.items():
            # Skip tokens not in profile
            if token not in allowed_tokens:
                continue
            
            while token in text:
                idx = text.index(token)
                line_no = text.count("\n", 0, idx) + 1
                val = gen()
                rel = p.relative_to(out_dir)
                
                # Use correct columns 
                secret_type = token.strip("{}").lower()  
                gt_rows.append({
                    "secret_type": secret_type,
                    "file_path": str(rel)
                })
                
                text = text.replace(token, val, 1)
                changed = True
        
        if changed:
            p.write_text(text, encoding="utf-8")
    
    # Write ground truth with correct columns
    gt_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(gt_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["secret_type", "file_path"])
        w.writeheader()
        w.writerows(gt_rows)
    
    print(f"✓ Profile: {profile}")
    print(f"✓ Injected {len(gt_rows)} secrets into {out_dir}")
    print(f"✓ Ground truth written to {gt_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default="payment_set_template")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gt", required=True)
    a = ap.parse_args()
    
    # Get profile from environment variable
    profile = os.getenv("INJECT_PROFILE", "medium").lower()
    print(f"Using injection profile: {profile}")
    
    inject(Path(a.template), Path(a.out), Path(a.gt), profile=profile)
