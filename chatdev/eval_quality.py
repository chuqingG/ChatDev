"""Lightweight evaluation utilities for ChatDev-generated projects.

This module exposes reusable functions so other scripts can import them without
triggering a full evaluation run on import. It keeps the original metrics
definitions (completeness, executability, consistency) but adds configurability
for API keys and output locations.
"""

import argparse
import os
import re
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from openai import OpenAI

DEFAULT_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-ada-002")


def _build_openai_client() -> Optional[OpenAI]:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("CHATDEV_OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE") or os.getenv("CHATDEV_OPENAI_API_BASE")
    if not api_key:
        return None
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


_CLIENT: Optional[OpenAI] = None


def get_client() -> Optional[OpenAI]:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = _build_openai_client()
    return _CLIENT


def get_files_from_type(source_dir: Path, filetype: str) -> list[Path]:
    files: list[Path] = []
    for root, _, filenames in os.walk(source_dir):
        for filename in filenames:
            if filename.endswith(filetype):
                files.append(Path(root) / filename)
    return files


def get_code(directory: Path) -> str:
    def _format_code(code: str) -> str:
        return "\n".join([line for line in code.split("\n") if line.strip()])

    codebooks = {}
    filepaths = get_files_from_type(directory, ".py")
    for filepath in filepaths:
        filename = filepath.name
        codebooks[filename] = _format_code(filepath.read_text(encoding="utf-8"))

    code = ""
    for filename in codebooks:
        code += f"{filename}\n```Python\n{codebooks[filename]}\n```\n\n"

    if not code:
        code = "# None"

    return code.strip()


def get_completeness(directory: Path) -> float:
    assert directory.is_dir()
    vn = get_code(directory)
    lines = vn.split("\n")
    lines = [
        line
        for line in lines
        if "pass" in line.lower() or "todo" in line.lower()
    ]
    lines = [
        line
        for line in lines
        if "password" not in line.lower()
        and "passenger" not in line.lower()
        and "passed" not in line.lower()
        and "passes" not in line.lower()
    ]
    if lines:
        return 0.0
    return 1.0


def _exist_bugs(directory: Path) -> tuple[bool, str, str]:
    assert directory.is_dir()
    success_info = "The software run successfully without errors."
    try:
        command = f"cd \"{directory}\"; ls -l; python3 main.py;"
        process = subprocess.Popen(
            command,
            shell=True,
            preexec_fn=os.setsid,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(3)

        error_type = ""
        return_code = process.returncode
        if process.poll() is None:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        if return_code == 0:
            return False, success_info, error_type
        error_output = process.stderr.read().decode("utf-8")
        try:
            error_pattern = r"\w+Error:"
            error_matches = re.findall(error_pattern, error_output)
            error_type = error_matches[0].replace(":", "")
        except Exception:
            pass
        if error_output and "traceback" in error_output.lower():
            errs = error_output.replace(str(directory) + "/", "")
            return True, errs, error_type
        return False, success_info, error_type
    except subprocess.CalledProcessError as exc:
        return True, f"Error: {exc}", "subprocess.CalledProcessError"
    except Exception as exc:  # noqa: BLE001
        return True, f"An error occurred: {exc}", "OtherException"


def _find_main_py_parent(directory: Path) -> Optional[Path]:
    for subroot, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(".py"):
                return Path(subroot)
    return None


def get_executability(directory: Path) -> float:
    pass_flag, _, _ = get_executability_detail(directory)
    if pass_flag:
        return 1.0
    return 0.0


def get_executability_detail(directory: Path) -> tuple[bool, str, str]:
    assert directory.is_dir()
    main_py_parent = _find_main_py_parent(directory)
    if main_py_parent is not None:
        bug_flag, info, error_type = _exist_bugs(main_py_parent)
        return (not bug_flag), info.replace("\n", "\\n"), error_type
    return False, "NoMain", "NoMain"


def _remove_comments(text: str) -> str:
    def _strip_regex(target: str, regex: str) -> str:
        lines = target.split("\n")
        lines = [line for line in lines if not line.strip().startswith("#")]
        target = "\n".join(lines)
        comments: list[str] = []
        matches = re.finditer(regex, target, re.DOTALL)
        for match in matches:
            comments.append(match.group(1))
        for comment in comments + ["''''''\n"]:
            target = target.replace(comment, "")
        return target

    text = _strip_regex(text, r"'''(.*?)'''")
    text = _strip_regex(text, r'"""(.*?)"""')
    return text


def _get_embedding(payload: str, client: OpenAI, model: str) -> list[float]:
    payload = payload if payload else "#"
    return client.embeddings.create(input=payload, model=model).model_dump()["data"][0]["embedding"]


def get_consistency(
    directory: Path,
    *,
    client: Optional[OpenAI] = None,
    model: str = DEFAULT_EMBED_MODEL,
    task_text: Optional[str] = None,
) -> Optional[float]:
    assert directory.is_dir()
    files = get_files_from_type(directory, ".txt")
    if files:
        task = files[0].read_text(encoding="utf-8").strip()
    else:
        task = (task_text or "").strip()
    if not task:
        return None
    codes = _remove_comments(get_code(directory))
    client = client or get_client()
    if client is None:
        return None
    text_embedding = _get_embedding(task, client, model)
    code_embedding = _get_embedding(codes, client, model)
    embeddingi = np.array(text_embedding)
    embeddingj = np.array(code_embedding)
    cos_sim = embeddingi.dot(embeddingj) / (np.linalg.norm(embeddingi) * np.linalg.norm(embeddingj))
    return float(cos_sim)


@dataclass
class EvalMetrics:
    project: str
    completeness: float
    executability: float
    consistency: Optional[float]
    exec_info: str = ""
    exec_error_type: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_project(
    directory: Path,
    *,
    task_text: Optional[str] = None,
    include_consistency: bool = True,
    client: Optional[OpenAI] = None,
    model: str = DEFAULT_EMBED_MODEL,
) -> EvalMetrics:
    completeness = get_completeness(directory)
    pass_flag, info, error_type = get_executability_detail(directory)
    executability = 1.0 if pass_flag else 0.0
    consistency = None
    if include_consistency:
        consistency = get_consistency(directory, client=client, model=model, task_text=task_text)
    return EvalMetrics(
        project=str(directory),
        completeness=completeness,
        executability=executability,
        consistency=consistency,
        exec_info=info,
        exec_error_type=error_type,
    )


def evaluate_warehouse(
    warehouse_root: Path,
    *,
    include_consistency: bool = True,
    client: Optional[OpenAI] = None,
    model: str = DEFAULT_EMBED_MODEL,
) -> list[EvalMetrics]:
    projects = sorted(p for p in warehouse_root.iterdir() if p.is_dir()) if warehouse_root.exists() else []
    results: list[EvalMetrics] = []
    for project in projects:
        results.append(
            evaluate_project(
                project,
                include_consistency=include_consistency,
                client=client,
                model=model,
            )
        )
    return results


def _write_tsv(path: Path, metrics: Iterable[EvalMetrics]) -> None:
    header = [
        "project",
        "completeness",
        "executability",
        "consistency",
        "exec_info",
        "exec_error_type",
    ]
    with path.open("w", encoding="utf-8") as writer:
        writer.write("\t".join(header) + "\n")
        for item in metrics:
            row = [
                item.project,
                f"{item.completeness:.4f}",
                f"{item.executability:.4f}",
                "" if item.consistency is None else f"{item.consistency:.6f}",
                item.exec_info,
                item.exec_error_type,
            ]
            writer.write("\t".join(row) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ChatDev WareHouse projects.")
    parser.add_argument("--warehouse-root", default="./WareHouse", help="Directory containing generated projects.")
    parser.add_argument("--output", default=None, help="TSV output path. Defaults to eval_quality.<root>.tsv next to this file.")
    parser.add_argument("--skip-consistency", action="store_true", help="Skip embedding-based consistency to avoid OpenAI calls.")
    args = parser.parse_args()

    warehouse_root = Path(args.warehouse_root).expanduser().resolve()
    client = None if args.skip_consistency else get_client()
    metrics = evaluate_warehouse(warehouse_root, include_consistency=not args.skip_consistency, client=client)

    if args.output:
        output_path = Path(args.output)
    else:
        suffix = str(warehouse_root).replace("/", "__").replace("-", "_")
        output_path = Path(__file__).with_suffix(f".{suffix}.tsv")
    _write_tsv(output_path, metrics)
    print(f"Evaluated {len(metrics)} project(s). Results saved to {output_path}.")


if __name__ == "__main__":
    main()
