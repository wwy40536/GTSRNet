"""Export installed package versions without credentials, URLs or local paths."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import platform
import re
from pathlib import Path


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("repro_env/server"))
    parser.add_argument("--overwrite", action="store_true", help="Replace an earlier export explicitly")
    args = parser.parse_args()
    destination = args.output_dir.expanduser().resolve()
    names = ("requirements-server.txt", "requirements-server-core.txt", "environment.json")
    existing = [str(destination / name) for name in names if (destination / name).exists()]
    if existing and not args.overwrite:
        parser.error("Export already exists; choose another directory or --overwrite: " + ", ".join(existing))

    packages = {}
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[normalize(name)] = {"name": name, "version": distribution.version}
    all_packages = sorted(packages.values(), key=lambda item: normalize(item["name"]))
    reference = Path(__file__).resolve().parents[1] / "requirements.txt"
    core_names = []
    if reference.is_file():
        for line in reference.read_text(encoding="utf-8-sig").splitlines():
            match = re.match(r"^([A-Za-z0-9_.-]+)==", line.strip())
            if match:
                core_names.append(normalize(match.group(1)))
    missing = [name for name in core_names if name not in packages]
    summary = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "packages": all_packages,
        "missing_reference_core_packages": missing,
        "note": "Actual installed versions; this is not a portable dependency lock or proof of completed experiments.",
    }
    try:
        import torch
    except Exception as error:
        summary["torch"] = {"import_ok": False, "error_type": type(error).__name__}
    else:
        summary["torch"] = {
            "import_ok": True,
            "version": str(torch.__version__),
            "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cudnn": torch.backends.cudnn.version(),
            "gpus": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        }
    header = "# Actual installed distribution versions; select the proper torch wheel index when reinstalling.\n"
    full_text = header + "".join(f"{item['name']}=={item['version']}\n" for item in all_packages)
    core_text = header + "".join(
        f"{packages[name]['name']}=={packages[name]['version']}\n" for name in core_names if name in packages
    )
    if missing:
        core_text += "# Not installed: " + ", ".join(missing) + "\n"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / names[0]).write_text(full_text, encoding="utf-8")
    (destination / names[1]).write_text(core_text, encoding="utf-8")
    (destination / names[2]).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Exported actual environment to", destination)
    print("Python:", summary["python"], "| installed distributions:", len(all_packages))
    if missing:
        print("Missing reference core packages:", ", ".join(missing))
    print("Package origins, tokens, environment variables and user paths were not exported.")


if __name__ == "__main__":
    main()
