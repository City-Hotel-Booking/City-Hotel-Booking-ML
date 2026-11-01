"""Package local project source/evidence while excluding data and fitted models."""
import hashlib
import json
import zipfile
from pathlib import Path


def package(root):
    root = Path(root)
    destination = root / "submission/clustering_source.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    excluded_dirs = {".git", "env", "venv", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "artifacts", "submission", ".idea", ".vscode", ".ipynb_checkpoints"}
    excluded_suffixes = {".joblib", ".pkl", ".pickle", ".onnx", ".h5", ".keras", ".pt", ".pth", ".pyc", ".log", ".docx", ".pdf"}
    files = []
    for top in ("src", "tests", "reports", "docs", "scripts", "app", "outputs", "examples", ".github"):
        folder = root / top
        if folder.exists():
            files.extend(p for p in folder.rglob("*") if p.is_file())
    files.extend(p for p in root.iterdir() if p.is_file())
    accepted = []
    for path in sorted(set(files)):
        rel = path.relative_to(root)
        if excluded_dirs.intersection(rel.parts) or path.suffix.lower() in excluded_suffixes:
            continue
        if rel.parts[:2] in (("outputs", "tmp"), ("outputs", "logs")) or path.name.startswith(".env"):
            continue
        if path.suffix.lower() not in {".py", ".md", ".txt", ".yaml", ".yml", ".json", ".csv", ".png", ".html", ".css", ".js", ""} and path.name not in {".gitignore", ".gitattributes"}:
            continue
        accepted.append((path, rel.as_posix()))
    manifest = {name: hashlib.sha256(path.read_bytes()).hexdigest() for path, name in accepted}
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path, name in accepted:
            archive.write(path, name)
        archive.writestr("clustering_source_manifest.json", json.dumps({"status": "local uncommitted clustering contribution; not an official submitted revision", "files": manifest}, indent=2))
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        assert not any(set(Path(n).parts) & {"artifacts", "env", ".venv", ".git"} or Path(n).parts[0] == "data" for n in archive.namelist())
        assert "src/clustering/complete.py" in archive.namelist()
        assert "outputs/clustering/m2/results.json" in archive.namelist()
    print(f"Verified {destination}: {len(accepted)} source/evidence files, {destination.stat().st_size:,} bytes; no raw data or fitted artifacts.")
    return destination


if __name__ == "__main__":
    package(Path(__file__).resolve().parents[1])
