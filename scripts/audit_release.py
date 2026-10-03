"""Conservative source audit. Findings report locations, never credential values."""

import argparse
import ipaddress
from pathlib import Path
import re
import subprocess

PATTERNS = {
    "internal mount": re.compile(r"/mnt/(?:bn|hdfs)/"),
    "private storage URI": re.compile(r"hdfs" r"://"),
    "internal service": re.compile(r"(?:byted(?:ance)?\.(?:org|net)|larkoffice\.com|sys[-]proxy)", re.IGNORECASE),
    "hardcoded credential": re.compile(
        r"(?:api[_-]?key|access[_-]?token|secret|password)\s*[:=]\s*['\"][A-Za-z0-9_+./=-]{16,}['\"]", re.IGNORECASE
    ),
    "credential token": re.compile(r"\b(?:hf_[A-Za-z0-9]{24,}|ghp_[A-Za-z0-9]{24,}|sk-[A-Za-z0-9]{24,})\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "credential URL": re.compile(r"https?://[^\s/@]+:[^\s/@]+@"),
}
IGNORED_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}


def source_files(root: Path) -> list[Path]:
    if (root / ".git").exists():
        result = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=root,
            check=True,
            capture_output=True,
        )
        return sorted({root / path.decode() for path in result.stdout.split(b"\0") if path})
    return sorted(path for path in root.rglob("*") if path.is_file() and not IGNORED_DIRS.intersection(path.parts))


def audit(root: Path) -> tuple[int, list[str]]:
    files = source_files(root)
    findings = []
    for path in files:
        relative = path.relative_to(root)
        if path.is_symlink():
            findings.append(f"{relative}: symlink (review destination before publication)")
            continue
        if path.stat().st_size > 5 * 1024**2:
            findings.append(f"{relative}: large file (possible data, model, or log artifact)")
            continue
        try:
            lines = path.read_text().splitlines()
        except UnicodeDecodeError:
            findings.append(f"{relative}: binary file requires manual review")
            continue
        for number, line in enumerate(lines, 1):
            for label, pattern in PATTERNS.items():
                if pattern.search(line):
                    findings.append(f"{relative}:{number}: {label}")
            # Check private IPv4 literals only in URL/host contexts, not version numbers.
            for match in re.finditer(r"(?:https?://|ws://|host\s*[:=]\s*['\"])((?:\d{1,3}\.){3}\d{1,3})", line):
                try:
                    address = ipaddress.ip_address(match.group(1))
                except ValueError:
                    continue
                if address.is_private and not (address.is_loopback or address.is_unspecified):
                    findings.append(f"{relative}:{number}: private IPv4 endpoint")
    return len(files), findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    count, findings = audit(args.root.resolve())
    for finding in findings:
        print(finding)
    print(f"Scanned {count} source files; {len(findings)} findings. Manual release review is still required.")
    raise SystemExit(bool(findings))


if __name__ == "__main__":
    main()
