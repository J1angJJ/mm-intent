from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def configure_stdio() -> None:
    """Force this wrapper's console streams to UTF-8."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(
            encoding="utf-8",
            errors="replace",
            line_buffering=True,
        )
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(
            encoding="utf-8",
            errors="replace",
            line_buffering=True,
        )


def parse_args(argv: list[str]) -> tuple[Path, list[str]]:
    try:
        log_index = argv.index("--log")
        separator_index = argv.index("--")
    except ValueError as exc:
        raise ValueError(
            "Usage: run_utf8_child.py --log LOG_FILE -- COMMAND [ARGS...]"
        ) from exc

    if log_index + 1 >= separator_index:
        raise ValueError("Missing log file path after --log.")

    log_path = Path(argv[log_index + 1]).resolve()
    command = argv[separator_index + 1:]

    if not command:
        raise ValueError("Missing child command after --.")

    return log_path, command


def main() -> int:
    configure_stdio()

    try:
        log_path, command = parse_args(sys.argv[1:])
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    log_path.parent.mkdir(parents=True, exist_ok=True)

    child_env = os.environ.copy()
    child_env["PYTHONUTF8"] = "1"
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env["PYTHONUNBUFFERED"] = "1"

    header = (
        f"COMMAND: {subprocess.list2cmdline(command)}\n"
        f"WORKING DIRECTORY: {Path.cwd()}\n"
        "TEXT ENCODING: UTF-8\n\n"
    )

    process: subprocess.Popen[str] | None = None

    with log_path.open(
        "w",
        encoding="utf-8",
        newline="",
        buffering=1,
    ) as log_file:
        sys.stdout.write(header)
        log_file.write(header)

        try:
            process = subprocess.Popen(
                command,
                cwd=Path.cwd(),
                env=child_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )

            if process.stdout is None:
                raise RuntimeError("Unable to capture child-process output.")

            for line in process.stdout:
                sys.stdout.write(line)
                log_file.write(line)

            return_code = process.wait()

        except KeyboardInterrupt:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

            message = "\nInterrupted by user.\n"
            sys.stdout.write(message)
            log_file.write(message)
            return 130

        footer = f"\nEXIT CODE: {return_code}\n"
        sys.stdout.write(footer)
        log_file.write(footer)

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
