# trivy/secrets_injector.py

import os, csv, shutil, base64, argparse, random, string, io
from pathlib import Path

# ---------- fake secret generators (all non-functional) ----------
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

TOKENS = {
    "{{AWS_ACCESS_KEY_ID}}": fake_aws_key,
    "{{AWS_SECRET_ACCESS_KEY}}": fake_aws_secret,
    "{{POSTGRES_URI}}": fake_postgres_uri,
    "{{GENERIC_API_KEY}}": fake_api_key,
    "{{DUMMY_JWT}}": fake_jwt,
}

# ---------- main ----------
def inject(template_dir: Path, out_dir: Path, gt_csv: Path):
    # copy the whole template tree first
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(template_dir, out_dir)
    
    gt_rows = []
    
    # walk every file and replace placeholders with fake values
    for p in out_dir.rglob("*"):
        if p.is_dir():
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        
        changed = False
        for token, gen in TOKENS.items():
            while token in text:
                idx = text.index(token)
                line_no = text.count("\n", 0, idx) + 1
                val = gen()
                rel = p.relative_to(out_dir)
                
                # FIX: Use correct column names for score_trivy.py
                secret_type = token.strip("{}").lower()  # Normalize to lowercase
                gt_rows.append({
                    "secret_type": secret_type,
                    "file_path": str(rel)
                })
                
                text = text.replace(token, val, 1)
                changed = True
        
        if changed:
            p.write_text(text, encoding="utf-8")
    
    # FIX: Write ground truth with correct column names
    gt_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(gt_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["secret_type", "file_path"])
        w.writeheader()
        w.writerows(gt_rows)
    
    print(f"✓ Injected {len(gt_rows)} secrets into {out_dir}")
    print(f"✓ Ground truth written to {gt_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default="payment_set_template")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gt", required=True)
    a = ap.parse_args()
    inject(Path(a.template), Path(a.out), Path(a.gt))
