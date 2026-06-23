#!/usr/bin/env python3

from __future__ import annotations

import fnmatch
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError


REPOS = [
    "duyle2408/results_dual_encoder_ablation_runs_v2",
    "duyle2408/report_runs_hybrid_tail_ablation",
    "duyle2408/results_machine3_fusion_loss",
    "duyle2408/report_runs_machine2_meta_aug_derm",
    "duyle2408/results_machine1_backbones",
    "duyle2408/report_single_encoder_no_freeze",
    "duyle2408/report_single_encoder",
]

IGNORE_PATTERNS = ["*.pt"]
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
REPORT_PATH = RESULTS_DIR / "download_report_rerun.json"


def list_remote_files(api: HfApi, repo_id: str, repo_type: str, token: str | None) -> list[str]:
    return api.list_repo_files(repo_id=repo_id, repo_type=repo_type, token=token)


def classify_error(exc: Exception) -> str:
    response = getattr(exc, "response", None)
    if response is not None and response.status_code in (401, 403):
        return "auth_required"
    return "download_failed"


def list_local_repo_files(destination: Path) -> list[Path]:
    return [
        path
        for path in destination.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(destination).parts
    ]


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    token = os.environ.get("HF_TOKEN")

    report: dict[str, object] = {"repos": [], "failures": []}

    for repo_id in REPOS:
        destination = RESULTS_DIR / repo_id.rsplit("/", 1)[-1]
        repo_record: dict[str, object] = {
            "repo_id": repo_id,
            "destination": str(destination),
            "remote_file_count": 0,
            "repo_type": None,
            "skipped_pt": [],
            "local_file_count": 0,
        }

        resolved = False
        last_error: Exception | None = None

        for repo_type in ("dataset", "model"):
            try:
                remote_files = list_remote_files(api, repo_id, repo_type, token)
                repo_record["repo_type"] = repo_type
                repo_record["remote_file_count"] = len(remote_files)
                repo_record["skipped_pt"] = [
                    path
                    for path in remote_files
                    if any(fnmatch.fnmatch(path, pattern) for pattern in IGNORE_PATTERNS)
                ]
                snapshot_download(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    token=token,
                    local_dir=destination,
                    local_dir_use_symlinks=False,
                    ignore_patterns=IGNORE_PATTERNS,
                )
                local_files = list_local_repo_files(destination)
                repo_record["local_file_count"] = len(local_files)
                report["repos"].append(repo_record)
                resolved = True
                break
            except RepositoryNotFoundError as exc:
                last_error = exc
                continue
            except HfHubHTTPError as exc:
                last_error = exc
                if getattr(exc, "response", None) is not None and exc.response.status_code in (401, 403):
                    report["failures"].append(
                        {
                            "repo_id": repo_id,
                            "destination": str(destination),
                            "error": classify_error(exc),
                            "message": str(exc),
                        }
                    )
                    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
                    return 2
                continue
            except Exception as exc:
                last_error = exc
                break

        if not resolved:
            report["failures"].append(
                {
                    "repo_id": repo_id,
                    "destination": str(destination),
                    "error": classify_error(last_error or RuntimeError("unknown error")),
                    "message": str(last_error) if last_error else "unknown error",
                }
            )

    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
