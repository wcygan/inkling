#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["rich>=15,<16", "rich-argparse>=1.7,<2"]
# ///
"""inkling: download the weights, build the GPU kernels, and run the server.

    uv run --script scripts/inkling.py doctor          check that the tools are installed, and how to install the missing ones
    uv run --script scripts/inkling.py status          what is ready and what to do next
    uv run --script scripts/inkling.py weights         download the checkpoint, then verify it
    uv run --script scripts/inkling.py check-weights   verify the checkpoint against the pinned hashes
    uv run --script scripts/inkling.py kernels         build the GPU kernels from source
    uv run --script scripts/inkling.py build           build the engine (cargo)
    uv run --script scripts/inkling.py serve           start (or replace) the server
    uv run --script scripts/inkling.py stop            stop the server

The commands are built so that they are hard to use wrongly. Each one checks what it needs and refuses, with the
next step, instead of failing later. The rules (checked with TLC, see docs/lifecycle) are:

  * Commands run one at a time: a lock file in logs/ is held from the first check to the end of a command, and the
    operating system releases it if a command is interrupted.
  * `serve` starts a server only with verified weights and kernels that match the current sources. It checks
    before it stops the running server and again right before it starts the new one, so there is never more than
    one server of this repository.
  * Nothing rewrites the weights or the kernels while a server is loading them.
  * "verified" and "fresh" are recorded last, and only if the files did not change while the command ran.

In a terminal the output has color, tables and a progress bar. Piped or redirected output, --json, and any run with
NO_COLOR set are plain text. FORCE_COLOR turns color on when the output is not a terminal.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

if sys.platform != "win32":
    sys.exit("inkling: this project runs on Windows x64 with an NVIDIA RTX 5090.")
import msvcrt  # noqa: E402  (Windows only)

ROOT = Path(__file__).resolve().parent.parent

# ---- pinned inputs -------------------------------------------------------------------------------------------
REPO = "thinkingmachines/Inkling-2-Mini"
REVISION = "7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6"
# Hugging Face's own hashes for that revision: 40 hex digits are git blob SHA-1, 64 are SHA-256 (large files).
# name -> (hash, size in bytes). README.md is not needed to run the model.
MANIFEST = {
    "chat_template.jinja": ("cae99e08917fe3d9ff21551a7c9a835981c69032", 6868),
    "config.json": ("e6c170c3fb666c50e3b509493b6b7a8f948b7d15", 2722),
    "generation_config.json": ("1b5d49d129001d2be58c2f254bfae3221eab6d96", 98),
    "hf_quant_config.json": ("ec915fa4d2b07bfe26240f71e981d34778022baf", 67924),
    "model-00001-of-00001.safetensors": ("65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1", 19526019168),
    "model.safetensors.index.json": ("3b835a38066039133931b604a0734dfae89a644e", 109300),
    "mtp.safetensors": ("7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2", 635801416),
    "processor_config.json": ("0e6df76a75a15f8af795dc08346dad6741f6421e", 461),
    "special_tokens_map.json": ("e4fe9735b528be1c630eb832d655fc8c4c2c309f", 529),
    "tiktoken/tokenizer.model": ("f170372dd655728909228220d77cf6b8104a59f38a91262a7d2c4027906ec0d8", 3386494),
    "tokenizer.json": ("7f332727f9a3f093dd452a39cc978e6757627e5dad1b0673aa4e75174f236dd6", 26178678),
    "tokenizer_config.json": ("abf8fb60451e6c33d2cfa3c4dfab9f8c1e7516aa", 13235),
}
OXIDE_REPO = "https://github.com/NVIDIA/cuda-rust"
OXIDE_COMMIT = "5e5c77b19a4b3204d6542bab2f0fde2a4eed824b"
KERNEL_FILES = ("inkling_kernels.cubin", "z_adaptive_select_width.cubin")
KERNEL_ARCH = "sm_120a"
MAX_CTX_LIMIT = 1_048_576  # the model's own limit
VRAM_GIB_AT_256K = 22.5  # measured on an RTX 5090 with FP8 KV, 6 checkpoints
VRAM_BYTES_PER_CTX_TOKEN = 16 * 1024  # FP8 KV cache, the model and the drafter

CONFIG_KEYS = {
    "max_ctx": int,
    "checkpoints": int,
    "effort": str,
    "port": int,
    "kv_fp8": bool,
    "draft_kv_fp8": bool,
    "model_dir": str,
    "pi_models_json": str,
}
EXIT_REFUSED = 2
BS = chr(92)  # a backslash, for f-strings


class Refused(Exception):
    """A command was stopped by a check before it changed anything. Carries what to do instead."""

    def __init__(self, reasons: list[str] | str):
        self.reasons = [reasons] if isinstance(reasons, str) else reasons
        super().__init__("; ".join(self.reasons))


class Failed(Exception):
    """A command started and could not finish."""


@dataclasses.dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def kernels(self) -> Path:
        return self.root / "kernels"

    @property
    def stamp(self) -> Path:
        return self.kernels / "build-stamp.json"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def lock(self) -> Path:
        return self.logs / "inkling.lock"

    @property
    def verified(self) -> Path:
        return self.logs / "weights-verified.json"

    @property
    def exe(self) -> Path:
        return self.root / "target" / "release" / "inkling.exe"

    @property
    def serve_exe(self) -> Path:
        """The server runs from a copy, so Windows never locks the build output and `build` works while it runs."""
        return self.root / "target" / "release" / "inkling-serve.exe"

    @property
    def oxide(self) -> Path:
        return self.root / "third_party" / "cuda-rust"

    @property
    def vsdev(self) -> Path:
        return self.root / "scripts" / "vsdev.cmd"

    @property
    def config(self) -> Path:
        return self.root / "serve.config.json"

    @property
    def vocab(self) -> Path:
        return self.root / "assets" / "draft_vocab_32k.bin"


P = Paths(ROOT)


# ---- output ----------------------------------------------------------------------------------------------------
# Color is for people. A command prints the same words either way; `rich` is imported only when it will be used.
GOOD, WARN, BAD, ACCENT = "green", "yellow", "bold red", "bold cyan"
_CONSOLES: dict = {}


def console():
    """A Rich console when the output should be styled, else None (plain text)."""
    if os.environ.get("NO_COLOR"):
        return None
    forced = bool(os.environ.get("FORCE_COLOR"))
    if not forced and (os.environ.get("TERM") == "dumb" or not (sys.stdout and sys.stdout.isatty())):
        return None
    if forced not in _CONSOLES:
        from rich.console import Console

        _CONSOLES[forced] = Console(force_terminal=True if forced else None, highlight=False, emoji=False, markup=False)
    return _CONSOLES[forced]


def say(*parts: str | tuple[str, str]) -> None:
    """Print one line made of parts: text, or (text, style). Brackets are printed as they are, and Rich does not break
    long lines, so commands stay whole when copied."""
    out = console()
    if out is None:
        print("".join(p if isinstance(p, str) else p[0] for p in parts), flush=True)
        return
    from rich.text import Text

    out.print(Text.assemble(*parts), soft_wrap=True)
    sys.stdout.flush()  # before a child process writes to the same terminal


def step(msg: str) -> None:
    say(("==> ", ACCENT), (msg, "bold"))


@contextlib.contextmanager
def hashing_progress(total: int):
    """A progress bar over `total` bytes in a terminal. Yields the function that advances it, or None in plain text."""
    out = console()
    if out is None:
        yield None
        return
    from rich.progress import BarColumn, DownloadColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn, TransferSpeedColumn

    columns = (TextColumn("   "), BarColumn(), TaskProgressColumn(), DownloadColumn(binary_units=True),
               TransferSpeedColumn(), TimeRemainingColumn())
    with Progress(*columns, console=out) as bar:
        task = bar.add_task("hashing", total=total)
        yield lambda n: bar.advance(task, n)


# ---- the command lock ----------------------------------------------------------------------------------------
class CommandLock:
    """One command at a time. A byte-range lock on logs/inkling.lock; Windows releases it if the process dies."""

    def __init__(self, paths: Paths, command: str):
        self.paths, self.command, self.file = paths, command, None

    def acquire(self, timeout: float = 900.0) -> None:
        self.paths.logs.mkdir(parents=True, exist_ok=True)
        self.file = open(self.paths.lock, "a+b")
        deadline, told = time.monotonic() + timeout, False
        while True:
            try:
                self.file.seek(0)
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                if not told:
                    say((f"Waiting for another inkling command to finish ({self.holder()})...", WARN))
                    told = True
                if time.monotonic() > deadline:
                    self.file.close()
                    self.file = None
                    raise Refused(f"another inkling command is still running ({self.holder()}); wait for it or stop it")
                time.sleep(0.1)
        info = json.dumps({"command": self.command, "pid": os.getpid()}).encode()
        self.file.seek(16)
        self.file.truncate(16)
        self.file.write(info)
        self.file.flush()

    def holder(self) -> str:
        try:
            with open(self.paths.lock, "rb") as f:
                f.seek(16)
                data = json.loads(f.read() or b"{}")
            return f"`{data.get('command', '?')}`, pid {data.get('pid', '?')}"
        except (OSError, ValueError):
            return "unknown"

    def release(self) -> None:
        if self.file is not None:
            with contextlib.suppress(OSError):
                self.file.seek(0)
                msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
            self.file.close()
            self.file = None


@contextlib.contextmanager
def command_lock(paths: Paths, command: str):
    lock = CommandLock(paths, command)
    lock.acquire()
    try:
        yield lock
    finally:
        lock.release()


# ---- weights -------------------------------------------------------------------------------------------------
def default_model_dir() -> Path:
    env = os.environ.get("INKLING_MODEL_DIR")
    return Path(env) if env else Path(os.environ.get("USERPROFILE", "~")).expanduser() / "Models" / "Inkling-2-Mini"


def _snapshot(f: Path) -> list[int]:
    st = f.stat()
    return [st.st_size, st.st_mtime_ns]


def weights_state(paths: Paths, model_dir: Path) -> tuple[str, list[str]]:
    """absent | partial | present (complete, not verified) | verified, with the reasons for the state."""
    present, problems = [], []
    for name, (_, size) in MANIFEST.items():
        f = model_dir / name
        if not f.is_file():
            problems.append(f"{name}: missing")
        elif f.stat().st_size != size:
            problems.append(f"{name}: {f.stat().st_size:,} bytes, expected {size:,}")
        else:
            present.append(name)
    if not present:
        return "absent", [f"no checkpoint files in {model_dir}"]
    partial_dir = model_dir / ".cache" / "huggingface" / "download"
    problems += [f"{p.name}: download in progress" for p in partial_dir.rglob("*.incomplete")] if partial_dir.is_dir() else []
    if problems:
        return "partial", problems
    try:
        marker = json.loads(paths.verified.read_text())
        ok = (
            marker.get("revision") == REVISION
            and marker.get("model_dir") == str(model_dir)
            and all(marker["files"].get(n) == _snapshot(model_dir / n) for n in MANIFEST)
        )
    except (OSError, ValueError, KeyError):
        ok = False
    if ok:
        return "verified", []
    return "present", ["the files have not been checked against the pinned hashes since they last changed"]


def _file_hash(f: Path, algo: str, size: int, advance=None) -> str:
    h = hashlib.sha256() if algo == "sha256" else hashlib.sha1(f"blob {size}\0".encode())
    buf = bytearray(8 << 20)  # one buffer for the whole file: allocating a new chunk per read costs a quarter of the time
    view = memoryview(buf)
    done, last = 0, time.monotonic()
    with open(f, "rb", buffering=0) as fh:
        while n := fh.readinto(buf):
            h.update(view[:n])
            done += n
            if advance:
                advance(n)
            elif size > (1 << 30) and time.monotonic() - last > 5:
                say(f"    {f.name}: {100 * done // size}%")
                last = time.monotonic()
    return h.hexdigest()


def verify_weights(paths: Paths, model_dir: Path) -> None:
    """Hash every file against the pinned revision. Records "verified" last, and only if nothing changed meanwhile."""
    state, why = weights_state(paths, model_dir)
    if state in ("absent", "partial"):
        raise Refused([f"the checkpoint is {state}: " + "; ".join(why[:4]), "download it first: `inkling weights`"])
    before = {n: _snapshot(model_dir / n) for n in MANIFEST}
    paths.verified.unlink(missing_ok=True)
    total = sum(size for _, size in MANIFEST.values())
    say(f"  hashing {len(MANIFEST)} files ({total / 2**30:.1f} GiB)")
    # the big file takes one thread and the small ones share the rest
    with hashing_progress(total) as advance, ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {name: pool.submit(_file_hash, model_dir / name, "sha256" if len(digest) == 64 else "sha1", size, advance)
                for name, (digest, size) in sorted(MANIFEST.items(), key=lambda kv: -kv[1][1])}
        got = {name: job.result() for name, job in jobs.items()}
    bad = [f"{name}: hash {got[name][:16]}..., expected {digest[:16]}..." for name, (digest, _) in MANIFEST.items() if got[name] != digest]
    if bad:
        raise Failed("these files differ from the pinned revision; download them again (`inkling weights --force`):\n  " + "\n  ".join(bad))
    after = {n: _snapshot(model_dir / n) for n in MANIFEST}
    if after != before:
        raise Failed("the weight files changed while they were being checked; run the command again")
    paths.logs.mkdir(parents=True, exist_ok=True)
    tmp = paths.verified.with_suffix(".tmp")
    tmp.write_text(json.dumps({"revision": REVISION, "model_dir": str(model_dir), "files": after}))
    os.replace(tmp, paths.verified)
    say((f"Checkpoint verified against revision {REVISION[:8]}.", GOOD))


# ---- kernels -------------------------------------------------------------------------------------------------
def kernel_sources(paths: Paths) -> list[Path]:
    r = paths.root
    files = [r / "crates/inkling-kernels/Cargo.toml", r / "crates/inkling-kernels/Cargo.lock", r / "crates/kernel-build/Cargo.toml",
             r / "rust-toolchain.toml", r / "patches/cuda-rust-windows-lean.patch"]
    for sub in ("src", "cuda"):
        d = r / "crates/inkling-kernels" / sub
        files += sorted(p for p in d.rglob("*") if p.is_file()) if d.is_dir() else []
    return [f for f in files if f.is_file()]


def sources_hash(paths: Paths) -> str:
    """Hash of everything the kernels are built from (line endings normalized, so a checkout does not change it)."""
    h = hashlib.sha256(f"{OXIDE_COMMIT}\0{KERNEL_ARCH}\0".encode())
    for f in kernel_sources(paths):
        h.update(f.relative_to(paths.root).as_posix().encode() + b"\0")
        h.update(f.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest()


def _sha256_file(f: Path) -> str:
    return hashlib.sha256(f.read_bytes()).hexdigest()


def kernel_state(paths: Paths) -> tuple[str, list[str]]:
    """absent | stale | fresh. Fresh means a stamp exists that matches the current sources and the files on disk."""
    files = [paths.kernels / n for n in KERNEL_FILES]
    if not any(f.is_file() for f in files):
        return "absent", ["the kernels have not been built"]
    if not all(f.is_file() for f in files):
        return "stale", ["a kernel file is missing"]
    try:
        stamp = json.loads(paths.stamp.read_text())
    except (OSError, ValueError):
        return "stale", ["no build record (an earlier build did not finish)"]
    if stamp.get("sources_sha256") != sources_hash(paths):
        return "stale", ["the kernel sources changed since the last build"]
    if any(stamp.get("files", {}).get(f.name) != _sha256_file(f) for f in files):
        return "stale", ["a kernel file differs from the build record"]
    return "fresh", []


def _run(cmd: list[str], cwd: Path | None = None, env: dict | None = None, quiet: bool = False) -> int:
    return subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.DEVNULL if quiet else None,
                          stderr=subprocess.STDOUT if quiet else None).returncode


def _stat_key(f: Path) -> list[int]:
    st = f.stat()
    return [st.st_mtime_ns, st.st_size]


def _vs_environment(paths: Paths) -> dict[str, str]:
    """The environment scripts/vsdev.cmd sets up: the Visual Studio C++ tools, CUDA, LLVM and vcpkg.

    vcvars64.bat alone takes about 2.5 s, so the result is cached in logs/vsenv.json and reused while everything it
    depends on is unchanged: vsdev.cmd, the variables it reads, PATH, and the Visual Studio files vcvars64.bat reads.
    Anything else is recomputed. Only the variables the setup changed are cached, so the rest of the environment is
    always the current one."""
    cache = paths.logs / "vsenv.json"
    inputs = {"vsdev": hashlib.sha256(paths.vsdev.read_bytes()).hexdigest(),
              "path": hashlib.sha256(os.environ.get("PATH", "").encode()).hexdigest(),
              **{k: os.environ.get(k, "") for k in ("CUDA_PATH", "CUDA_TOOLKIT_PATH", "LIBCLANG_PATH", "VCPKG_ROOT")}}
    try:
        data = json.loads(cache.read_text())
        if data["inputs"] == inputs and all(Path(f).exists() and _stat_key(Path(f)) == key for f, key in data["watch"].items()):
            env = dict(os.environ)
            env.update(data["set"])
            env["PATH"] = data["path"]
            return env
    except (OSError, ValueError, KeyError):
        pass
    r = subprocess.run(["cmd", "/c", str(paths.vsdev), "set"], capture_output=True, text=True)
    if r.returncode != 0:
        raise Failed("could not set up the Visual Studio C++ environment (scripts\\vsdev.cmd); run `inkling doctor`")
    captured: dict[str, str] = {}
    for line in r.stdout.splitlines():
        name, sep, value = line.partition("=")
        if sep and name and not name.startswith("="):
            captured[name.upper()] = value
    parent = {k.upper(): v for k, v in os.environ.items()}
    changed = {k: v for k, v in captured.items() if k != "PATH" and parent.get(k) != v}
    build_dir = Path(captured.get("VCINSTALLDIR", "")) / "Auxiliary" / "Build"
    kits = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Windows Kits" / "10"  # vcvars picks the newest SDK in here
    watch = {str(f): _stat_key(f) for f in (paths.vsdev, build_dir / "vcvars64.bat", build_dir / "Microsoft.VCToolsVersion.default.txt",
                                            kits / "Include", kits / "Lib") if f.exists()}
    new_path = captured.get("PATH") or os.environ.get("PATH", "")
    paths.logs.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    tmp.write_text(json.dumps({"inputs": inputs, "watch": watch, "set": changed, "path": new_path}))
    os.replace(tmp, cache)
    env = dict(os.environ)
    env.update(changed)
    env["PATH"] = new_path
    return env


def _vs(paths: Paths, *cmd: str, cwd: Path | None = None, quiet: bool = False) -> int:
    """Run a command inside the MSVC and CUDA environment that scripts/vsdev.cmd sets up (cached, see above)."""
    env = _vs_environment(paths)
    exe = shutil.which(cmd[0], path=env.get("PATH")) or cmd[0]
    return _run([exe, *cmd[1:]], cwd=cwd, env=env, quiet=quiet)


def _run_captured(cmd: list[str], env: dict) -> tuple[int, str]:
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


# ---- doctor: what must be installed, and how to install what is missing ------------------------------------
STAGES = ("run", "weights", "build", "kernels")  # what the tools are needed for
CUDA_MIN = (13, 5)
DISK_NEEDED_GIB = 22  # the checkpoint (about 20 GB) and some room


@dataclasses.dataclass
class Check:
    key: str
    title: str
    needed_for: tuple[str, ...]  # a subset of STAGES; optional checks may be empty
    ok: bool
    detail: str
    fix: str = ""
    optional: bool = False


def _out(cmd: list[str], timeout: int = 30) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (r.stdout + r.stderr).strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def vs_install() -> Path | None:
    """The Visual Studio 2022 (or Build Tools) install that has the C++ tools, found with vswhere."""
    pf86 = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
    vswhere = pf86 / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if vswhere.is_file():
        out = _out([str(vswhere), "-latest", "-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-property", "installationPath"])
        if out and (Path(out) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat").is_file():
            return Path(out)
    return None


def cuda_dir() -> Path:
    return Path(os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5"))


def cublas_dir() -> Path | None:
    """The folder with cublas64_13.dll: on PATH, or in the CUDA Toolkit."""
    for d in [*os.environ.get("PATH", "").split(os.pathsep), str(cuda_dir() / "bin" / "x64"), str(cuda_dir() / "bin")]:
        if d and (Path(d) / "cublas64_13.dll").is_file():
            return Path(d)
    return None


def _on_path(d: Path) -> bool:
    return os.path.normcase(str(d)) in [os.path.normcase(x.rstrip("\\")) for x in os.environ.get("PATH", "").split(os.pathsep)]


def _free_gib(path: Path) -> float:
    p = path
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free / 2**30


def gpu_check() -> Check:
    with ThreadPoolExecutor(max_workers=2) as pool:
        job_info, job_smi = pool.submit(gpu_info), pool.submit(_out, ["nvidia-smi"])
        gpu, smi = job_info.result(), job_smi.result()
    dm = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", smi) if gpu else None
    drv = (int(dm.group(1)), int(dm.group(2))) if dm else None
    if not gpu:
        ok, detail = False, "nvidia-smi found no NVIDIA GPU or driver"
    elif gpu["compute_cap"] != "12.0":
        ok = False
        detail = f"{gpu['name']} has compute capability {gpu['compute_cap']}; the kernels are built for {KERNEL_ARCH} (RTX 5090 class, 12.0)"
    elif drv and drv < CUDA_MIN:
        ok, detail = False, f"{gpu['name']}; the driver supports CUDA {drv[0]}.{drv[1]} and 13.5 is needed"
    else:
        ok = True
        detail = f"{gpu['name']}, {gpu['total_gib']:.0f} GiB" + (f", driver supports CUDA {drv[0]}.{drv[1]}" if drv else "")
    return Check("gpu", "NVIDIA RTX 5090 and driver", ("run", "kernels"), ok, detail,
                 "install the current NVIDIA driver from https://www.nvidia.com/drivers (an RTX 5090, compute capability 12.0, is required)")


def cublas_check() -> Check:
    cb = cublas_dir()
    if cb is None:
        return Check("cublas", "cuBLAS 13 (cublas64_13.dll)", ("run",), False, "not found on PATH or in the CUDA Toolkit",
                     "install the CUDA Toolkit (see above); it includes cuBLAS")
    return Check("cublas", "cuBLAS 13 (cublas64_13.dll)", ("run",), True,
                 f"{cb}" + ("" if _on_path(cb) else " (not on PATH; `inkling serve` adds it for the server)"))


def os_check(paths: Paths, model_dir: Path) -> Check:
    ok = platform.machine().upper() in ("AMD64", "X86_64")
    return Check("os", "Windows x64", STAGES, ok, f"{platform.system()} {platform.release()} {platform.machine()}",
                 "" if ok else "this project runs on Windows x64 only")


def uv_check(paths: Paths, model_dir: Path) -> Check:
    uv = shutil.which("uv")
    return Check("uv", "uv", ("weights",), bool(uv), (_out([uv, "--version"]) if uv else "not found"), "winget install --id astral-sh.uv -e")


def git_check(paths: Paths, model_dir: Path) -> Check:
    git = shutil.which("git")
    return Check("git", "git", ("kernels",), bool(git), (_out([git, "--version"]) if git else "not found"), "winget install --id Git.Git -e")


def rust_check(paths: Paths, model_dir: Path) -> Check:
    rustup, cargo = shutil.which("rustup"), shutil.which("cargo")
    toolchain = paths.root / "rust-toolchain.toml"
    pinned = re.search(r'channel\s*=\s*"([^"]+)"', toolchain.read_text()) if toolchain.is_file() else None
    channel = pinned.group(1) if pinned else "the pinned toolchain"
    if rustup and cargo:
        # the installed toolchains are folders under RUSTUP_HOME; asking `rustup toolchain list` costs a process for the same answer
        home = Path(os.environ.get("RUSTUP_HOME") or Path.home() / ".rustup") / "toolchains"
        have = any(channel in d.name for d in home.iterdir()) if home.is_dir() else channel in _out([rustup, "toolchain", "list"])
        detail = f"rustup found; Rust {channel} " + ("is installed" if have else "installs itself on the first build (needs network)")
    else:
        detail = "not found"
    return Check("rust", "Rust (rustup)", ("build", "kernels"), bool(rustup and cargo), detail, "winget install --id Rustlang.Rustup -e")


def msvc_check(paths: Paths, model_dir: Path) -> Check:
    vs = vs_install()
    return Check("msvc", "Visual Studio C++ tools", ("build", "kernels"), vs is not None, str(vs) if vs else "not found",
                 'winget install --id Microsoft.VisualStudio.2022.BuildTools -e --override "--passive --wait '
                 '--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"')


def cuda_check(paths: Paths, model_dir: Path) -> Check:
    nvcc = cuda_dir() / "bin" / "nvcc.exe"
    m = re.search(r"release (\d+)\.(\d+)", _out([str(nvcc), "--version"])) if nvcc.is_file() else None
    ver = (int(m.group(1)), int(m.group(2))) if m else None
    ok = ver is not None and ver >= CUDA_MIN
    if ok:
        detail = f"CUDA {ver[0]}.{ver[1]} at {cuda_dir()}"
    elif ver:
        detail = f"CUDA {ver[0]}.{ver[1]} found at {cuda_dir()}; 13.5 or newer is needed"
    else:
        detail = f"nvcc not found in {cuda_dir()}"
    return Check("cuda", "CUDA Toolkit 13.5 (nvcc)", ("kernels",), ok, detail,
                 "winget install --id Nvidia.CUDA --version 13.5 -e   (if the installer offers a display driver, keep your current one; "
                 "if CUDA is installed elsewhere, set CUDA_PATH)")


def llvm_check(paths: Paths, model_dir: Path) -> Check:
    clang = Path(os.environ.get("LIBCLANG_PATH", r"C:\Program Files\LLVM\bin"))
    has = (clang / "libclang.dll").is_file()
    return Check("llvm", "LLVM (libclang)", ("kernels",), has, str(clang) if has else f"libclang.dll not found in {clang}",
                 "winget install --id LLVM.LLVM -e   (if LLVM is installed elsewhere, set LIBCLANG_PATH)")


def libffi_check(paths: Paths, model_dir: Path) -> Check:
    vcpkg = Path(os.environ.get("VCPKG_ROOT", r"C:\vcpkg"))
    ffi = list((vcpkg / "installed" / "x64-windows" / "lib").glob("ffi*.lib"))
    return Check("libffi", "libffi (vcpkg)", ("kernels",), bool(ffi), f"found under {vcpkg}" if ffi else f"not found under {vcpkg}",
                 f"git clone https://github.com/microsoft/vcpkg {vcpkg}; {vcpkg}{BS}bootstrap-vcpkg.bat -disableMetrics; "
                 f"{vcpkg}{BS}vcpkg.exe install libffi:x64-windows   (if vcpkg is elsewhere, set VCPKG_ROOT)")


def disk_check(paths: Paths, model_dir: Path) -> Check:
    w, _ = weights_state(paths, model_dir)
    if w in ("absent", "partial"):
        free = _free_gib(model_dir)
        return Check("disk", f"{DISK_NEEDED_GIB} GiB of free disk for the weights", ("weights",), free >= DISK_NEEDED_GIB,
                     f"{free:.0f} GiB free on the drive of {model_dir}",
                     "free some space, or put the checkpoint on another drive with --model-dir or `model_dir` in serve.config.json")
    return Check("disk", "disk space for the weights", ("weights",), True, "the checkpoint is already downloaded")


def just_check(paths: Paths, model_dir: Path) -> Check:
    have = bool(shutil.which("just"))
    return Check("just", "just (shortcuts)", (), have, "found" if have else "not found; use `uv run --script scripts/inkling.py ...` instead",
                 "winget install --id Casey.Just -e", optional=True)


def pi_check(paths: Paths, model_dir: Path) -> Check:
    have = bool(shutil.which("pi"))
    return Check("pi", "Pi coding agent", (), have, "found" if have else "not found",
                 "npm i -g @mariozechner/pi-coding-agent   (needs Node.js)", optional=True)


# (probe, the stages that need it, optional): in the order to install the tools
PROBES = (
    (os_check, STAGES, False), (uv_check, ("weights",), False), (git_check, ("kernels",), False),
    (rust_check, ("build", "kernels"), False), (msvc_check, ("build", "kernels"), False), (cuda_check, ("kernels",), False),
    (llvm_check, ("kernels",), False), (libffi_check, ("kernels",), False),
    (lambda paths, model_dir: gpu_check(), ("run", "kernels"), False), (lambda paths, model_dir: cublas_check(), ("run",), False),
    (disk_check, ("weights",), False), (just_check, (), True), (pi_check, (), True),
)


def run_checks(paths: Paths, model_dir: Path | None = None, stages: tuple[str, ...] | None = None, optional: bool = True) -> list[Check]:
    """The tools the commands need, in the order to install them. `stages` limits it to what those stages need, and
    `optional` says whether to include the optional tools. The probes are independent, so they run side by side."""
    model_dir = model_dir or default_model_dir()
    chosen = [fn for fn, needed, opt in PROBES
              if (not opt or optional) and (stages is None or any(s in needed for s in stages) or opt)]
    with ThreadPoolExecutor(max_workers=max(1, len(chosen))) as pool:
        return list(pool.map(lambda fn: fn(paths, model_dir), chosen))


def failing(checks: list[Check], stage: str) -> list[Check]:
    return [c for c in checks if not c.ok and not c.optional and stage in c.needed_for]


def _describe(c: Check) -> str:
    return f"{c.title}: {c.detail}" + (f". Install: {c.fix}" if c.fix else "")


def refuse_if_missing(paths: Paths, stage: str, model_dir: Path | None = None) -> None:
    bad = failing(run_checks(paths, model_dir, stages=(stage,), optional=False), stage)
    if bad:
        raise Refused([_describe(c) for c in bad] + ["`inkling doctor` checks everything and lists the commands"])


def kernel_prerequisites() -> list[str]:
    return [_describe(c) for c in failing(run_checks(P, stages=("kernels",), optional=False), "kernels")]


def _doctor_report(checks: list[Check], stage: str) -> tuple[list[tuple[str, str, Check]], list[tuple[str, str]], int]:
    """What `doctor` reports: (mark, needed, check) rows, the closing (line, style) pairs, and the exit code."""
    shown = [c for c in checks if stage == "all" or stage in c.needed_for or c.optional]
    def needed(c: Check) -> str:
        if c.optional:
            return "optional"
        stages = [s for s in c.needed_for if stage in ("all", s)]
        return "needed for everything" if len(stages) == len(STAGES) else "needed for " + ", ".join(stages)

    rows = [("ok" if c.ok else ("skip" if c.optional else "MISSING"), needed(c), c) for c in shown]
    missing = [c for c in shown if not c.ok and not c.optional]
    optional = [c for c in shown if not c.ok and c.optional]
    closing = []
    if missing:
        closing.append((f"{len(missing)} required item(s) missing. Install them in this order (PowerShell; some installers ask for approval):", BAD))
        closing += [(f"  {i}. {c.fix}", "cyan") for i, c in enumerate(missing, 1) if c.fix]
        closing.append(("Then open a NEW terminal, so the installs are on PATH, and run `inkling doctor` again.", WARN))
    else:
        closing.append(("Everything needed is installed." if stage == "all" else f"Everything needed for `{stage}` is installed.",
                        f"bold {GOOD}"))
    if optional:
        closing.append(("Optional: " + "; ".join(f"{c.title}: {c.fix}" for c in optional), "dim"))
    return rows, closing, (EXIT_REFUSED if missing else 0)


def render_doctor(checks: list[Check], stage: str) -> tuple[list[str], int]:
    """The plain-text report."""
    rows, closing, code = _doctor_report(checks, stage)
    width, nwidth = max(len(c.title) for _, _, c in rows), max(len(needed) for _, needed, _ in rows)
    lines = [f"  [{mark:<7}] {c.title:<{width}}  {needed:<{nwidth}}  {c.detail}" for mark, needed, c in rows]
    return lines + [""] + [line for line, _ in closing], code


def show_doctor(checks: list[Check], stage: str) -> int:
    """The same report as a table, for a terminal."""
    from rich import box
    from rich.table import Table
    from rich.text import Text

    rows, closing, code = _doctor_report(checks, stage)
    styles = {"ok": GOOD, "skip": "dim", "MISSING": BAD}
    table = Table(box=box.SIMPLE_HEAD, header_style="bold", pad_edge=False)
    table.add_column("status", no_wrap=True)
    table.add_column("tool")
    table.add_column("needed for")
    table.add_column("detail", overflow="fold")
    for mark, needed, c in rows:
        table.add_row(Text(mark, style=styles[mark]), c.title, Text(needed.removeprefix("needed for "), style="dim"), c.detail)
    console().print(table)
    for line in closing:
        say(line)
    return code


def build_kernels(paths: Paths) -> None:
    missing = kernel_prerequisites()
    if missing:
        raise Refused(["missing prerequisites:"] + [f"  - {m}" for m in missing] + ["install them and run `inkling kernels` again"])
    # 1. cuda-oxide at the pinned commit, with the small Windows fix
    if not (paths.oxide / "Cargo.toml").is_file():
        step(f"cloning {OXIDE_REPO} into third_party/cuda-rust")
        paths.oxide.parent.mkdir(parents=True, exist_ok=True)
        if _run(["git", "clone", "--quiet", OXIDE_REPO, str(paths.oxide)]) != 0:
            raise Failed("git clone failed")
    head = subprocess.run(["git", "-C", str(paths.oxide), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != OXIDE_COMMIT:
        step(f"checking out cuda-rust {OXIDE_COMMIT[:9]}")
        if _run(["git", "-C", str(paths.oxide), "checkout", "--quiet", OXIDE_COMMIT]) != 0:
            raise Failed("git checkout failed; remove third_party/cuda-rust and run again")
    patch = paths.root / "patches" / "cuda-rust-windows-lean.patch"
    if subprocess.run(["git", "-C", str(paths.oxide), "apply", "--ignore-whitespace", "--check", str(patch)], capture_output=True).returncode == 0:
        step("applying patches/cuda-rust-windows-lean.patch")
        if _run(["git", "-C", str(paths.oxide), "apply", "--ignore-whitespace", str(patch)]) != 0:
            raise Failed("applying the cuda-rust patch failed")
    # 2. cargo-oxide and its codegen backend
    if not shutil.which("cargo-oxide"):
        step("installing cargo-oxide")
        if _run(["cargo", "install", "--locked", "--path", str(paths.oxide / "cuda-oxide" / "crates" / "cargo-oxide")], cwd=paths.root) != 0:
            raise Failed("installing cargo-oxide failed")
    if _vs(paths, "cargo", "oxide", "doctor", cwd=paths.root, quiet=True) != 0:
        step("building the cuda-oxide codegen backend (`cargo oxide setup`, a few minutes the first time)")
        if _vs(paths, "cargo", "oxide", "setup", cwd=paths.root) != 0 or _vs(paths, "cargo", "oxide", "doctor", cwd=paths.root) != 0:
            raise Failed("`cargo oxide setup` or `doctor` failed; read the output above")
    # 3. build. The record that marks the kernels fresh is deleted first and written last.
    paths.stamp.unlink(missing_ok=True)
    start_hash = sources_hash(paths)
    out = paths.root / "crates" / "kernel-build" / "kernels"
    shutil.rmtree(out, ignore_errors=True)
    paths.kernels.mkdir(exist_ok=True)
    env = _vs_environment(paths)  # once, before the two compilers start
    cu = paths.root / "crates" / "inkling-kernels" / "cuda" / "z_adaptive_select_width.cu"
    nvcc_tmp = paths.kernels / "z_adaptive_select_width.cubin.tmp"
    nvcc_cmd = [shutil.which("nvcc", path=env.get("PATH")) or "nvcc", "-O3", "-std=c++17", "-gencode", f"arch=compute_120a,code={KERNEL_ARCH}",
                "-cubin", str(cu), "-o", str(nvcc_tmp)]
    step(f"compiling crates/inkling-kernels for {KERNEL_ARCH} (cargo oxide build) and z_adaptive_select_width.cu (nvcc), side by side")
    with ThreadPoolExecutor(max_workers=1) as pool:
        nvcc_job = pool.submit(_run_captured, nvcc_cmd, env)
        oxide_rc = _vs(paths, "cargo", "oxide", "build", "--arch", KERNEL_ARCH, cwd=paths.root / "crates" / "kernel-build")
        nvcc_rc, nvcc_text = nvcc_job.result()
    if oxide_rc != 0:
        nvcc_tmp.unlink(missing_ok=True)
        raise Failed("`cargo oxide build` failed; read the output above")
    if nvcc_rc != 0:
        print(nvcc_text, flush=True)  # the compiler's own text, as it wrote it
        raise Failed("nvcc failed; its output is above")
    built = out / "inkling_kernels.cubin"
    if not built.is_file():
        raise Failed(f"cargo oxide reported success but {built} does not exist")
    shutil.copyfile(built, paths.kernels / "inkling_kernels.cubin.tmp")
    for name in KERNEL_FILES:
        os.replace(paths.kernels / f"{name}.tmp", paths.kernels / name)
    # 4. stamp last, and only if the sources did not change while building
    if sources_hash(paths) != start_hash:
        raise Failed("the kernel sources changed while building; the kernels are not marked fresh. Run `inkling kernels` again")
    paths.stamp.write_text(json.dumps({"sources_sha256": start_hash, "oxide_commit": OXIDE_COMMIT,
                                       "files": {n: _sha256_file(paths.kernels / n) for n in KERNEL_FILES}}, indent=1))
    for n in KERNEL_FILES:
        say("  ", (f"{n:<34}", "cyan"), f" {(paths.kernels / n).stat().st_size:>10,} bytes  ",
            (f"sha256 {_sha256_file(paths.kernels / n)[:16]}", "dim"))


# ---- the server ----------------------------------------------------------------------------------------------
def _powershell_json(script: str):
    r = subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, text=True, timeout=60)
    out = r.stdout.strip()
    if not out:
        return []
    data = json.loads(out)
    return data if isinstance(data, list) else [data]


_WIN32 = None


def _win32():
    """kernel32 with the signatures this tool uses (loaded on first use)."""
    global _WIN32
    if _WIN32 is None:
        import ctypes
        from ctypes import wintypes

        class Entry(ctypes.Structure):
            _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                        ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                        ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", wintypes.LONG), ("dwFlags", wintypes.DWORD),
                        ("szExeFile", wintypes.WCHAR * 260)]

        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        k.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        k.Process32FirstW.argtypes = k.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(Entry)]
        k.OpenProcess.restype = wintypes.HANDLE
        k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
        k.CloseHandle.argtypes = [wintypes.HANDLE]
        _WIN32 = (ctypes, wintypes, k, Entry)
    return _WIN32


def _processes(names: set[str]) -> list[tuple[int, str]]:
    """(pid, image name) of the running processes whose image name is in `names`: one process snapshot, a few milliseconds."""
    ctypes, wintypes, k, Entry = _win32()
    snap = k.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
    if not snap or snap == ctypes.c_void_p(-1).value:
        return []
    found = []
    try:
        entry = Entry()
        entry.dwSize = ctypes.sizeof(Entry)
        ok = k.Process32FirstW(snap, ctypes.byref(entry))
        while ok:
            if entry.szExeFile.lower() in names:
                found.append((entry.th32ProcessID, entry.szExeFile))
            ok = k.Process32NextW(snap, ctypes.byref(entry))
    finally:
        k.CloseHandle(snap)
    return found


def _image_path(pid: int) -> str | None:
    """The full path of a running process's executable, or None if it is not readable."""
    ctypes, wintypes, k, _ = _win32()
    handle = k.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return None
    try:
        buf = ctypes.create_unicode_buffer(32768)
        size = wintypes.DWORD(len(buf))
        return buf.value if k.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)) else None
    finally:
        k.CloseHandle(handle)


def our_servers(paths: Paths) -> list[dict]:
    """Running servers of this repository: `inkling-serve.exe` (the copy `serve` runs) and `inkling.exe serve` (a server an
    older version started). Other copies of inkling are left alone."""
    mine = {os.path.normcase(str(p.resolve())) for p in (paths.exe, paths.serve_exe)}
    found, legacy = [], []
    for pid, name in _processes({"inkling.exe", "inkling-serve.exe"}):
        path = _image_path(pid)
        if not path or os.path.normcase(path) not in mine:
            continue
        if name.lower() == "inkling-serve.exe":
            found.append({"ProcessId": pid, "ExecutablePath": path, "CommandLine": None})
        else:
            legacy.append(pid)
    if legacy:  # only an `inkling.exe serve` counts, which needs its command line: rare, so the slow query is fine
        rows = _powershell_json("Get-CimInstance Win32_Process -Filter \"" + " OR ".join(f"ProcessId={p}" for p in legacy) + "\" | "
                                "Select-Object ProcessId,ExecutablePath,CommandLine | ConvertTo-Json -Compress")
        found += [r for r in rows if re.search(r"\sserve(\s|$)", r.get("CommandLine") or "")]
    return found


def port_owner(port: int) -> dict | None:
    out = subprocess.run(["netstat", "-ano", "-p", "tcp"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        cols = line.split()
        if len(cols) >= 5 and cols[3] == "LISTENING" and cols[1].rsplit(":", 1)[-1] == str(port):
            return {"ProcessId": int(cols[4]), "ExecutablePath": _image_path(int(cols[4]))}
    return None


def health(port: int) -> int | None:
    """200 ready, 503 loading, None nothing listening."""
    import http.client

    try:
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        c.request("GET", "/health")
        return c.getresponse().status
    except OSError:
        return None


def server_state(paths: Paths, port: int) -> str:
    if not our_servers(paths):
        return "stopped"
    return "ready" if health(port) == 200 else "loading"


def stop_servers(paths: Paths, port: int) -> int:
    procs = our_servers(paths)
    for r in procs:
        say((f"Stopping the server (pid {r['ProcessId']}).", WARN))
        subprocess.run(["taskkill", "/PID", str(r["ProcessId"]), "/T", "/F"], capture_output=True)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and (our_servers(paths) or (procs and port_owner(port))):
        time.sleep(0.1)
    if our_servers(paths):
        raise Failed("the running server did not stop")
    return len(procs)


def gpu_info() -> dict:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap,memory.total,memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout.strip().splitlines()[0]
        name, cc, total, free = [x.strip() for x in out.split(",")]
        return {"name": name, "compute_cap": cc, "total_gib": int(total) / 1024, "free_gib": int(free) / 1024}
    except (OSError, IndexError, ValueError, subprocess.TimeoutExpired):
        return {}


@dataclasses.dataclass
class Settings:
    max_ctx: int = 262144
    checkpoints: int = 6
    effort: str = "low"
    port: int = 8013
    kv_fp8: bool = True
    draft_kv_fp8: bool = True
    model_dir: Path = dataclasses.field(default_factory=default_model_dir)
    pi_models_json: str = ""

    def validate(self) -> list[str]:
        bad = []
        if not 1024 <= self.max_ctx <= MAX_CTX_LIMIT:
            bad.append(f"max_ctx {self.max_ctx} must be between 1024 and {MAX_CTX_LIMIT:,}")
        if not 0 <= self.checkpoints <= 32:
            bad.append(f"checkpoints {self.checkpoints} must be between 0 and 32")
        if not 1024 <= self.port <= 65535:
            bad.append(f"port {self.port} must be between 1024 and 65535")
        if not re.fullmatch(r"[a-z0-9.]{1,16}", self.effort):
            bad.append(f"effort {self.effort!r} must be a short lowercase word or number, such as low or high")
        return bad


def load_settings(paths: Paths, args: argparse.Namespace) -> Settings:
    s = Settings()
    if paths.config.is_file():
        try:
            cfg = json.loads(paths.config.read_text())
        except ValueError as e:
            raise Refused(f"{paths.config.name} is not valid JSON: {e}")
        unknown = sorted(set(cfg) - set(CONFIG_KEYS))
        if unknown:
            raise Refused(f"{paths.config.name}: unknown setting(s) {unknown}; allowed: {sorted(CONFIG_KEYS)}")
        for key, value in cfg.items():
            if not isinstance(value, CONFIG_KEYS[key]) or (CONFIG_KEYS[key] is int and isinstance(value, bool)):
                raise Refused(f"{paths.config.name}: {key} must be {CONFIG_KEYS[key].__name__}, not {value!r}")
            if key == "model_dir":
                if value:
                    s.model_dir = Path(value)
            else:
                setattr(s, key, value)
    for key in ("max_ctx", "checkpoints", "effort", "port", "kv_fp8", "draft_kv_fp8", "pi_models_json"):
        if getattr(args, key, None) is not None:
            setattr(s, key, getattr(args, key))
    if getattr(args, "model_dir", None):
        s.model_dir = Path(args.model_dir)
    problems = s.validate()
    if problems:
        raise Refused(problems)
    return s


def serve_checks(paths: Paths, s: Settings, unverified_ok: bool) -> list[str]:
    """Why `serve` cannot start now. Empty means it can. Runs before the old server is stopped, and again just before the start."""
    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs = (pool.submit(weights_state, paths, s.model_dir), pool.submit(kernel_state, paths), pool.submit(gpu_check),
                pool.submit(cublas_check), pool.submit(port_owner, s.port), pool.submit(our_servers, paths))
        (w, wwhy), (k, kwhy), gpu_c, cublas_c, owner, ours = (j.result() for j in jobs)
    why = []
    if w in ("absent", "partial"):
        why.append(f"the checkpoint is {w} ({'; '.join(wwhy[:3])}). Download it: `inkling weights`")
    elif w == "present" and not unverified_ok:
        why.append("the checkpoint has not been verified. Run `inkling check-weights` (or pass --unverified-weights to skip the check)")
    if k != "fresh":
        why.append(f"the GPU kernels are {k} ({kwhy[0]}). Build them: `inkling kernels`")
    why += [_describe(c) for c in (gpu_c, cublas_c) if not c.ok]
    if owner and not any(r["ProcessId"] == owner["ProcessId"] for r in ours):
        why.append(f"port {s.port} is used by {owner.get('ExecutablePath') or 'another program'} (pid {owner['ProcessId']}), which is not this repository's server. Stop it or choose another --port")
    return why


def vram_warning(s: Settings) -> str | None:
    gpu = gpu_info()
    if not gpu:
        return None
    need = VRAM_GIB_AT_256K - (262144 - s.max_ctx) * VRAM_BYTES_PER_CTX_TOKEN / 2**30
    if gpu["free_gib"] + 0.5 < need:
        return f"about {need:.1f} GiB of VRAM is needed for max_ctx {s.max_ctx:,} and {gpu['free_gib']:.1f} GiB is free (close other GPU programs, or lower --max-ctx)"
    return None


def sync_pi(path: str, port: int, max_ctx: int) -> None:
    f = Path(path.replace("~", str(Path.home()), 1))
    if not f.is_file():
        return
    cfg = json.loads(f.read_text(encoding="utf-8-sig"))
    prov = cfg.get("providers", {}).get("inkling")
    if not prov:
        return
    prov["baseUrl"] = f"http://127.0.0.1:{port}/v1"
    for m in prov.get("models", []):
        m["contextWindow"] = max_ctx
    f.write_text(json.dumps(cfg, indent=2), encoding="utf-8")  # no BOM: Pi parses this with JSON.parse
    say((f"Updated {f}: inkling contextWindow {max_ctx:,}, port {port}.", GOOD))


def build_host(paths: Paths) -> None:
    refuse_if_missing(paths, "build")
    step("building the engine (cargo build --release --locked -p inkling-cli)")
    if _vs(paths, "cargo", "build", "--release", "--locked", "-p", "inkling-cli", cwd=paths.root) != 0 or not paths.exe.is_file():
        raise Failed("the engine did not build; read the output above")


def wait_ready(port: int, proc: subprocess.Popen, timeout: float = 300.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise Failed(f"the server exited with code {proc.returncode} while loading; its output is above")
        if health(port) == 200:
            return
        time.sleep(0.1)
    raise Failed(f"the server was not ready after {timeout:.0f} s")


# ---- commands ------------------------------------------------------------------------------------------------
def cmd_status(args: argparse.Namespace) -> int:
    s = load_settings(P, args)
    with ThreadPoolExecutor(max_workers=5) as pool:
        jobs = (pool.submit(weights_state, P, s.model_dir), pool.submit(kernel_state, P), pool.submit(server_state, P, s.port),
                pool.submit(gpu_info), pool.submit(run_checks, P, s.model_dir))
        (w, wwhy), (k, kwhy), srv, gpu, checks = (j.result() for j in jobs)
    exe = P.exe.is_file()
    missing = [c for c in checks if not c.ok and not c.optional]
    if args.json:
        print(json.dumps({"weights": w, "kernels": k, "engine": exe, "server": srv, "missing_tools": [c.key for c in missing],
                          "model_dir": str(s.model_dir), "gpu": gpu}, indent=1), flush=True)
        return 0
    todo = []
    if missing:
        todo.append("inkling doctor")
    if w in ("absent", "partial"):
        todo.append("inkling weights")
    elif w == "present":
        todo.append("inkling check-weights")
    if k != "fresh":
        todo.append("inkling kernels")
    if not exe:
        todo.append("inkling build")
    nxt = ", then ".join(todo) if todo else ("inkling serve" if srv == "stopped" else "ready to use")
    # (label, state, the state's style, detail)
    rows = [
        ("weights", w, GOOD if w == "verified" else BAD if w == "absent" else WARN, f"{s.model_dir}" + (f"  ({wwhy[0]})" if wwhy else "")),
        ("kernels", k, GOOD if k == "fresh" else BAD if k == "absent" else WARN, f"({kwhy[0]})" if kwhy else ""),
        ("engine", "built" if exe else "not built", GOOD if exe else BAD, ""),
        ("server", srv, GOOD if srv == "ready" else WARN, f"http://127.0.0.1:{s.port}/v1" if srv != "stopped" else ""),
        ("gpu", "found" if gpu else "not found", GOOD if gpu else BAD,
         f"{gpu['name']}  {gpu['free_gib']:.1f} of {gpu['total_gib']:.1f} GiB free" if gpu else ""),
    ]
    if missing:
        rows.append(("setup", "incomplete", BAD, f"{len(missing)} required tool(s) missing: " + ", ".join(c.title for c in missing)))
    if console():
        show_status(rows, nxt)
    else:
        for line in render_status(rows, nxt):
            say(line)
    return 0


def render_status(rows: list[tuple[str, str, str, str]], nxt: str) -> list[str]:
    """The plain-text report. A state is padded to 9 characters where a detail follows it."""
    lines = []
    for label, state, _, detail in rows:
        if label == "setup":
            text = detail
        elif label == "gpu":  # the name says it was found
            text = detail or state
        elif label == "weights":
            text = f"{state:<9} {detail}"
        elif label == "kernels":  # padded even when nothing follows
            text = f"{state:<9}" + (f"  {detail}" if detail else "")
        else:
            text = f"{state:<9} {detail}" if detail else state
        lines.append(f"{label:<10}{text}")
    return lines + [f"{'next':<10}{nxt}"]


def show_status(rows: list[tuple[str, str, str, str]], nxt: str) -> None:
    """The same report as a panel, for a terminal."""
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold", no_wrap=True)
    table.add_column()
    table.add_column(overflow="fold")
    for label, state, style, detail in rows:
        table.add_row(label, Text(state, style=style), detail)
    console().print(Panel(table, title="inkling", title_align="left", border_style="cyan", expand=False))
    say(("next  ", "bold"), (nxt, f"bold {GOOD}" if nxt == "ready to use" else ACCENT))


def cmd_doctor(args: argparse.Namespace) -> int:
    s = load_settings(P, args)
    # --json lists every tool whatever the stage, as it always has; the text report shows only the stage's
    checks = run_checks(P, s.model_dir, stages=None if (args.stage == "all" or args.json) else (args.stage,))
    if args.json:
        print(json.dumps([dataclasses.asdict(c) for c in checks], indent=1), flush=True)
        return 0 if not [c for c in checks if not c.ok and not c.optional and (args.stage == "all" or args.stage in c.needed_for)] else EXIT_REFUSED
    if console():
        return show_doctor(checks, args.stage)
    lines, code = render_doctor(checks, args.stage)
    for line in lines:
        say(line)
    return code


def cmd_weights(args: argparse.Namespace) -> int:
    s = load_settings(P, args)
    with command_lock(P, "weights"):
        state, _ = weights_state(P, s.model_dir)
        if state == "verified" and not args.force:
            say(("The checkpoint is already downloaded and verified. Use --force to download it again.", GOOD))
            return 0
        if state in ("absent", "partial") or args.force:
            if our_servers(P):  # only a download rewrites files; checking them is safe while a server runs
                raise Refused("a server is running; stop it first (`inkling stop`) so the files are not changed under it")
            refuse_if_missing(P, "weights", s.model_dir)
            uv = shutil.which("uv")
            if not uv:
                raise Refused("uv is needed to run the Hugging Face downloader (winget install astral-sh.uv)")
            step(f"downloading {REPO} at {REVISION[:8]} into {s.model_dir} (about 20 GB; an interrupted download resumes)")
            P.verified.unlink(missing_ok=True)
            cmd = [uv, "tool", "run", "--from", "huggingface_hub", "hf", "download", REPO, "--revision", REVISION, "--local-dir", str(s.model_dir)]
            if _run(cmd) != 0:
                raise Failed("the download did not finish. Run `inkling weights` again to resume. If the repository asks you to accept "
                             "its terms, run `uv tool run --from huggingface_hub hf auth login` first")
        step("verifying the download")
        verify_weights(P, s.model_dir)
    return 0


def cmd_check_weights(args: argparse.Namespace) -> int:
    s = load_settings(P, args)
    with command_lock(P, "check-weights"):
        verify_weights(P, s.model_dir)
    return 0


def cmd_kernels(args: argparse.Namespace) -> int:
    with command_lock(P, "kernels"):
        if kernel_state(P)[0] == "fresh" and not args.force:
            say(("The kernels are already built from the current sources. Use --force to build them again.", GOOD))
            return 0
        build_kernels(P)
        say(("Kernels built.", GOOD))
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    with command_lock(P, "build"):
        build_host(P)
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    s = load_settings(P, args)
    with command_lock(P, "stop"):
        n = stop_servers(P, s.port)
    say(("Server stopped.", GOOD) if n else "No server of this repository is running.")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    s = load_settings(P, args)
    lock = CommandLock(P, "serve")
    lock.acquire()
    try:
        why = serve_checks(P, s, args.unverified_weights)
        if why:
            raise Refused(why)
        build_host(P)
        argv = [str(P.serve_exe), "serve", "--mode", "fp8", "--draft-vocab", str(P.vocab), "--max-ctx", str(s.max_ctx),
                "--effort", s.effort, "--model-dir", str(s.model_dir)]
        env = dict(os.environ, INKLING_PORT=str(s.port), INKLING_CHECKPOINTS=str(s.checkpoints),
                   INKLING_SERVE_LOG=str(P.logs / "serve-log.jsonl"))
        cb = cublas_dir()
        if cb and not _on_path(cb):
            env["PATH"] = f"{cb}{os.pathsep}{env.get('PATH', '')}"
        for name, on in (("INKLING_KV_FP8", s.kv_fp8), ("INKLING_DRAFT_KV_FP8", s.draft_kv_fp8)):
            env.pop(name, None)
            if on:
                env[name] = "1"
        if args.dry_run:
            warn = None if our_servers(P) else vram_warning(s)  # a running server's memory is freed by the replacement
            if warn:
                say(("Warning: ", f"bold {WARN}"), (warn, WARN))
            say(("Dry run: everything checks out.", GOOD), " Would stop this repository's server if one is running, then start:")
            say(("  " + " ".join(argv), "cyan"))
            return 0
        stop_servers(P, s.port)
        shutil.copyfile(P.exe, P.serve_exe)  # after the old server stopped, so the copy is not locked
        why = serve_checks(P, s, args.unverified_weights)  # again, right before the start
        if why:
            raise Refused(["the checks changed while the old server was stopping:"] + why)
        warn = vram_warning(s)  # now that the old server has released its memory
        if warn:
            say(("Warning: ", f"bold {WARN}"), (warn, WARN))
        if s.pi_models_json:
            sync_pi(s.pi_models_json, s.port, s.max_ctx)
        step(f"starting the server (max_ctx {s.max_ctx:,}, port {s.port})")
        proc = subprocess.Popen(argv, cwd=P.root, env=env)
        try:
            wait_ready(s.port, proc)
        except Failed:
            proc.terminate()
            raise
        except KeyboardInterrupt:
            proc.terminate()
            return 130
        lock.release()  # loading is done; other commands may run now
        say(("Server ready: ", f"bold {GOOD}"), (f"http://127.0.0.1:{s.port}/v1", ACCENT),
            "  (model inkling-2-mini). Ctrl+C stops it.")
        try:
            code = proc.wait()
            if code == 1:  # what Windows reports for a process that was force-stopped
                say(("The server was stopped (from another terminal, or replaced by a newer `serve`).", WARN))
                return 0
            return code
        except KeyboardInterrupt:
            proc.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=30)
            return 0
    finally:
        lock.release()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="inkling", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser, model_dir: bool = True) -> None:
        if model_dir:
            p.add_argument("--model-dir", help="checkpoint folder (default: serve.config.json, then INKLING_MODEL_DIR, then %%USERPROFILE%%\\Models\\Inkling-2-Mini)")

    p = sub.add_parser("status", help="show what is ready and what to do next")
    common(p)
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_status)
    p = sub.add_parser("doctor", help="check that everything needed is installed, and say how to install what is missing")
    common(p)
    p.add_argument("--for", dest="stage", choices=("all", *STAGES), default="all", help="check only what one stage needs")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_doctor)
    p = sub.add_parser("weights", help="download the checkpoint at the pinned revision, then verify it")
    common(p)
    p.add_argument("--force", action="store_true", help="download again even if the checkpoint is verified")
    p.set_defaults(fn=cmd_weights)
    p = sub.add_parser("check-weights", help="verify the checkpoint against the pinned hashes")
    common(p)
    p.set_defaults(fn=cmd_check_weights)
    p = sub.add_parser("kernels", help="build the GPU kernels from source into kernels/")
    p.add_argument("--force", action="store_true", help="build again even if the kernels are fresh")
    p.set_defaults(fn=cmd_kernels)
    p = sub.add_parser("build", help="build the engine with cargo")
    p.set_defaults(fn=cmd_build)
    p = sub.add_parser("stop", help="stop this repository's server")
    common(p)
    p.add_argument("--port", type=int)
    p.set_defaults(fn=cmd_stop)
    p = sub.add_parser("serve", help="start the server, replacing this repository's running one")
    common(p)
    p.add_argument("--max-ctx", type=int, help="context length in tokens")
    p.add_argument("--checkpoints", type=int, help="saved cache states for turn reuse")
    p.add_argument("--effort", help="default reasoning effort")
    p.add_argument("--port", type=int)
    p.add_argument("--kv-fp8", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--draft-kv-fp8", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--unverified-weights", action="store_true", help="allow a complete checkpoint that has not been verified")
    p.add_argument("--dry-run", action="store_true", help="run every check and print the command, but start and stop nothing")
    p.set_defaults(fn=cmd_serve)
    if console():
        for parser in (ap, *sub.choices.values()):
            parser.formatter_class = _color_help(parser.formatter_class)
    return ap


def _color_help(plain: type) -> type:
    """argparse's help in color, with the headings and the text as argparse writes them."""
    import rich_argparse

    raw = plain is argparse.RawDescriptionHelpFormatter
    base = rich_argparse.RawDescriptionRichHelpFormatter if raw else rich_argparse.RichHelpFormatter

    class Help(base):
        group_name_formatter = str
        help_markup = text_markup = False

    return Help


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except Refused as e:
        say(("inkling: not done.", BAD))
        for r in e.reasons:
            if not r.startswith("  ") and not r.endswith(":"):
                say(("  - ", BAD), r)
            else:
                say("  ", (r, WARN if r.endswith(":") else ""))
        return EXIT_REFUSED
    except Failed as e:
        say(("inkling: failed: ", BAD), str(e))
        return 1
    except KeyboardInterrupt:
        say(("inkling: interrupted.", WARN))
        return 130


if __name__ == "__main__":
    sys.exit(main())
