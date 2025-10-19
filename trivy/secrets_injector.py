# trivy/secrets_injector.py
import os, csv, shutil, base64, argparse, random, string
from pathlib import Path

def fake_aws_key():    return "AKIA" + "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ2345671", k=16))
def fake_aws_secret(): return base64.b64encode(os.urandom(30)).decode()[:40]
def fake_postgres_uri():
    user = "user_" + ''.join(random.choices(string.ascii_lowercase, k=5))
    pwd  = base64.b16encode(os.urandom(8)).decode().lower()
    return f"postgres://{user}:{pwd}@db.example.com:5432/app"
def fake_api_key():    return "sk_test_" + base64.b32encode(os.urandom(12)).decode().strip("=")
def fake_jwt():
    import base64 as b64
    return "eyJhbGciOiJIUzI1NiJ9." + b64.urlsafe_b64encode(os.urandom(18)).decode().strip("=") + "." + b64.urlsafe_b64encode(os.urandom(18)).decode().strip("=")

TOKENS = {
    "{{AWS_ACCESS_KEY_ID}}": fake_aws_key,
    "{{AWS_SECRET_ACCESS_KEY}}": fake_aws_secret,
    "{{POSTGRES_URI}}": fake_postgres_uri,
    "{{GENERIC_API_KEY}}": fake_api_key,
    "{{DUMMY_JWT}}": fake_jwt,
}

def inject(template_dir: Path, out_dir: Path, gt_csv: Path, extra_per_file: int = 0):
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(template_dir, out_dir)

    gt_rows, rule_id = [], 1

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
                gt_rows.append({
                    "type": token.strip("{}"),
                    "file": str(rel),
                    "start_line": line_no,
                    "end_line": line_no,
                    "rule_id": f"secret_rule_{rule_id}"
                })
                rule_id += 1
                text = text.replace(token, val, 1)
                changed = True

        if extra_per_file > 0:
            for k in range(extra_per_file):
                token, gen = random.choice(list(TOKENS.items()))
                val = gen()
                # append at EOF: KEY=VALUE
                appended = f"{token.strip('{}')}={val}\n"
                current_lines = text.count("\n") + 1
                start_line = end_line = current_lines + 1
                rel = p.relative_to(out_dir)
                gt_rows.append({
                    "type": token.strip("{}"),
                    "file": str(rel),
                    "start_line": start_line,
                    "end_line": end_line,
                    "rule_id": f"secret_rule_append_{k+1}"
                })
                text += appended
                changed = True

        if changed:
            p.write_text(text, encoding="utf-8")

    gt_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(gt_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["type","file","start_line","end_line","rule_id"])
        w.writeheader(); w.writerows(gt_rows)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default="payment_set_template")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--extra-per-file", type=int, default=0, help="append N extra fake secrets per text file")
    a = ap.parse_args()
    inject(Path(a.template), Path(a.out), Path(a.gt), a.extra_per_file)
