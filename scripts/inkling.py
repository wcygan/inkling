#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""inkling: download the weights, build the GPU kernels, and run the server.

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
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import http.client
import json
import os
import re
import shutil
import subprocess
import sys
import time
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


def say(msg: str = "") -> None:
    print(msg, flush=True)


def step(msg: str) -> None:
    say(f"==> {msg}")


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
                    say(f"Waiting for another inkling command to finish ({self.holder()})...")
                    told = True
                if time.monotonic() > deadline:
                    self.file.close()
                    self.file = None
                    raise Refused(f"another inkling command is still running ({self.holder()}); wait for it or stop it")
                time.sleep(0.5)
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


def _file_hash(f: Path, algo: str, size: int) -> str:
    h = hashlib.sha256() if algo == "sha256" else hashlib.sha1(f"blob {size}\0".encode())
    done, last = 0, time.monotonic()
    with open(f, "rb") as fh:
        while chunk := fh.read(8 << 20):
            h.update(chunk)
            done += len(chunk)
            if size > (1 << 30) and time.monotonic() - last > 5:
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
    bad = []
    for name, (digest, size) in MANIFEST.items():
        say(f"  hashing {name}")
        got = _file_hash(model_dir / name, "sha256" if len(digest) == 64 else "sha1", size)
        if got != digest:
            bad.append(f"{name}: hash {got[:16]}..., expected {digest[:16]}...")
    if bad:
        raise Failed("these files differ from the pinned revision; download them again (`inkling weights --force`):\n  " + "\n  ".join(bad))
    after = {n: _snapshot(model_dir / n) for n in MANIFEST}
    if after != before:
        raise Failed("the weight files changed while they were being checked; run the command again")
    paths.logs.mkdir(parents=True, exist_ok=True)
    tmp = paths.verified.with_suffix(".tmp")
    tmp.write_text(json.dumps({"revision": REVISION, "model_dir": str(model_dir), "files": after}))
    os.replace(tmp, paths.verified)
    say(f"Checkpoint verified against revision {REVISION[:8]}.")


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


def _vs(paths: Paths, *cmd: str, cwd: Path | None = None, quiet: bool = False) -> int:
    """Run a command inside the MSVC and CUDA environment set up by scripts/vsdev.cmd."""
    return _run(["cmd", "/c", str(paths.vsdev), *cmd], cwd=cwd, quiet=quiet)


def kernel_prerequisites() -> list[str]:
    missing = []
    for tool, how in (("git", "winget install Git.Git"), ("cargo", "install Rust from https://rustup.rs"), ("uv", "winget install astral-sh.uv")):
        if not shutil.which(tool):
            missing.append(f"{tool} ({how})")
    cuda = Path(os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5"))
    if not (cuda / "bin" / "nvcc.exe").is_file():
        missing.append(f"the CUDA 13.5 Toolkit with nvcc (looked in {cuda}; set CUDA_PATH if it is elsewhere)")
    clang = Path(os.environ.get("LIBCLANG_PATH", r"C:\Program Files\LLVM\bin"))
    if not (clang / "libclang.dll").is_file():
        missing.append(f"LLVM with libclang.dll (winget install LLVM.LLVM; looked in {clang}; set LIBCLANG_PATH if it is elsewhere)")
    vcpkg = Path(os.environ.get("VCPKG_ROOT", r"C:\vcpkg"))
    if not list((vcpkg / "installed" / "x64-windows" / "lib").glob("ffi*.lib")):
        missing.append(f"libffi from vcpkg (vcpkg install libffi:x64-windows; looked under {vcpkg}; set VCPKG_ROOT if it is elsewhere)")
    editions = ("Community", "BuildTools", "Professional", "Enterprise")
    if not any((Path(r"C:\Program Files\Microsoft Visual Studio\2022") / e / "VC/Auxiliary/Build/vcvars64.bat").is_file() for e in editions):
        missing.append("Visual Studio 2022 with the C++ workload (scripts/vsdev.cmd calls its vcvars64.bat)")
    return missing


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
    step(f"compiling crates/inkling-kernels for {KERNEL_ARCH} (cargo oxide build)")
    if _vs(paths, "cargo", "oxide", "build", "--arch", KERNEL_ARCH, cwd=paths.root / "crates" / "kernel-build") != 0:
        raise Failed("`cargo oxide build` failed; read the output above")
    paths.kernels.mkdir(exist_ok=True)
    built = out / "inkling_kernels.cubin"
    if not built.is_file():
        raise Failed(f"cargo oxide reported success but {built} does not exist")
    step("compiling z_adaptive_select_width.cu with nvcc")
    cu = paths.root / "crates" / "inkling-kernels" / "cuda" / "z_adaptive_select_width.cu"
    nvcc_out = paths.kernels / "z_adaptive_select_width.cubin.tmp"
    if _vs(paths, "nvcc", "-O3", "-std=c++17", "-gencode", f"arch=compute_120a,code={KERNEL_ARCH}", "-cubin", str(cu), "-o", str(nvcc_out)) != 0:
        raise Failed("nvcc failed; read the output above")
    shutil.copyfile(built, paths.kernels / "inkling_kernels.cubin.tmp")
    for name in KERNEL_FILES:
        os.replace(paths.kernels / f"{name}.tmp", paths.kernels / name)
    # 4. stamp last, and only if the sources did not change while building
    if sources_hash(paths) != start_hash:
        raise Failed("the kernel sources changed while building; the kernels are not marked fresh. Run `inkling kernels` again")
    paths.stamp.write_text(json.dumps({"sources_sha256": start_hash, "oxide_commit": OXIDE_COMMIT,
                                       "files": {n: _sha256_file(paths.kernels / n) for n in KERNEL_FILES}}, indent=1))
    for n in KERNEL_FILES:
        say(f"  {n:<34} {(paths.kernels / n).stat().st_size:>10,} bytes  sha256 {_sha256_file(paths.kernels / n)[:16]}")


# ---- the server ----------------------------------------------------------------------------------------------
def _powershell_json(script: str):
    r = subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, text=True, timeout=60)
    out = r.stdout.strip()
    if not out:
        return []
    data = json.loads(out)
    return data if isinstance(data, list) else [data]


def our_servers(paths: Paths) -> list[dict]:
    """Running `inkling.exe serve` processes of this repository (other copies of inkling are left alone)."""
    rows = _powershell_json("Get-CimInstance Win32_Process -Filter \"Name='inkling.exe'\" | "
                            "Select-Object ProcessId,ExecutablePath,CommandLine | ConvertTo-Json -Compress")
    mine = os.path.normcase(str(paths.exe.resolve()))
    return [r for r in rows if r.get("ExecutablePath") and os.path.normcase(r["ExecutablePath"]) == mine
            and re.search(r"\sserve(\s|$)", r.get("CommandLine") or "")]


def port_owner(port: int) -> dict | None:
    out = subprocess.run(["netstat", "-ano", "-p", "tcp"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        cols = line.split()
        if len(cols) >= 5 and cols[3] == "LISTENING" and cols[1].rsplit(":", 1)[-1] == str(port):
            rows = _powershell_json(f"Get-CimInstance Win32_Process -Filter \"ProcessId={int(cols[4])}\" | "
                                    "Select-Object ProcessId,ExecutablePath | ConvertTo-Json -Compress")
            return rows[0] if rows else {"ProcessId": int(cols[4]), "ExecutablePath": None}
    return None


def health(port: int) -> int | None:
    """200 ready, 503 loading, None nothing listening."""
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
        say(f"Stopping the server (pid {r['ProcessId']}).")
        subprocess.run(["taskkill", "/PID", str(r["ProcessId"]), "/T", "/F"], capture_output=True)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and (our_servers(paths) or (procs and port_owner(port))):
        time.sleep(0.5)
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
    why = []
    w, wwhy = weights_state(paths, s.model_dir)
    if w in ("absent", "partial"):
        why.append(f"the checkpoint is {w} ({'; '.join(wwhy[:3])}). Download it: `inkling weights`")
    elif w == "present" and not unverified_ok:
        why.append("the checkpoint has not been verified. Run `inkling check-weights` (or pass --unverified-weights to skip the check)")
    k, kwhy = kernel_state(paths)
    if k != "fresh":
        why.append(f"the GPU kernels are {k} ({kwhy[0]}). Build them: `inkling kernels`")
    gpu = gpu_info()
    if not gpu:
        why.append("nvidia-smi found no NVIDIA GPU or driver")
    elif gpu["compute_cap"] != "12.0":
        why.append(f"the kernels are built for {KERNEL_ARCH} (RTX 5090 class); this GPU is {gpu['name']} (compute capability {gpu['compute_cap']})")
    owner = port_owner(s.port)
    if owner and not any(r["ProcessId"] == owner["ProcessId"] for r in our_servers(paths)):
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
    say(f"Updated {f}: inkling contextWindow {max_ctx:,}, port {port}.")


def build_host(paths: Paths) -> None:
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
        time.sleep(0.5)
    raise Failed(f"the server was not ready after {timeout:.0f} s")


# ---- commands ------------------------------------------------------------------------------------------------
def cmd_status(args: argparse.Namespace) -> int:
    s = load_settings(P, args)
    w, wwhy = weights_state(P, s.model_dir)
    k, kwhy = kernel_state(P)
    srv = server_state(P, s.port)
    gpu = gpu_info()
    exe = P.exe.is_file()
    if args.json:
        say(json.dumps({"weights": w, "kernels": k, "engine": exe, "server": srv, "model_dir": str(s.model_dir), "gpu": gpu}, indent=1))
        return 0
    say(f"weights   {w:<9} {s.model_dir}" + (f"  ({wwhy[0]})" if wwhy else ""))
    say(f"kernels   {k:<9}" + (f"  ({kwhy[0]})" if kwhy else ""))
    say(f"engine    {'built' if exe else 'not built'}")
    say(f"server    {srv:<9} http://127.0.0.1:{s.port}/v1" if srv != "stopped" else "server    stopped")
    say(f"gpu       {gpu.get('name', 'not found')}" + (f"  {gpu['free_gib']:.1f} of {gpu['total_gib']:.1f} GiB free" if gpu else ""))
    todo = []
    if w in ("absent", "partial"):
        todo.append("inkling weights")
    elif w == "present":
        todo.append("inkling check-weights")
    if k != "fresh":
        todo.append("inkling kernels")
    if not exe:
        todo.append("inkling build")
    say("next      " + (", then ".join(todo) if todo else ("inkling serve" if srv == "stopped" else "ready to use")))
    return 0


def cmd_weights(args: argparse.Namespace) -> int:
    s = load_settings(P, args)
    with command_lock(P, "weights"):
        if our_servers(P):
            raise Refused("a server is running; stop it first (`inkling stop`) so the files are not changed under it")
        state, _ = weights_state(P, s.model_dir)
        if state == "verified" and not args.force:
            say("The checkpoint is already downloaded and verified. Use --force to download it again.")
            return 0
        if state in ("absent", "partial") or args.force:
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
            say("The kernels are already built from the current sources. Use --force to build them again.")
            return 0
        build_kernels(P)
        say("Kernels built.")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    with command_lock(P, "build"):
        build_host(P)
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    s = load_settings(P, args)
    with command_lock(P, "stop"):
        n = stop_servers(P, s.port)
    say("Server stopped." if n else "No server of this repository is running.")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    s = load_settings(P, args)
    lock = CommandLock(P, "serve")
    lock.acquire()
    try:
        why = serve_checks(P, s, args.unverified_weights)
        if why:
            raise Refused(why)
        warn = vram_warning(s)
        if warn:
            say(f"Warning: {warn}")
        build_host(P)
        argv = [str(P.exe), "serve", "--mode", "fp8", "--draft-vocab", str(P.vocab), "--max-ctx", str(s.max_ctx),
                "--effort", s.effort, "--model-dir", str(s.model_dir)]
        env = dict(os.environ, INKLING_PORT=str(s.port), INKLING_CHECKPOINTS=str(s.checkpoints),
                   INKLING_SERVE_LOG=str(P.logs / "serve-log.jsonl"))
        for name, on in (("INKLING_KV_FP8", s.kv_fp8), ("INKLING_DRAFT_KV_FP8", s.draft_kv_fp8)):
            env.pop(name, None)
            if on:
                env[name] = "1"
        if args.dry_run:
            say("Dry run: everything checks out. Would stop this repository's server if one is running, then start:")
            say("  " + " ".join(argv))
            return 0
        stop_servers(P, s.port)
        why = serve_checks(P, s, args.unverified_weights)  # again, right before the start
        if why:
            raise Refused(["the checks changed while the old server was stopping:"] + why)
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
        say(f"Server ready: http://127.0.0.1:{s.port}/v1  (model inkling-2-mini). Ctrl+C stops it.")
        try:
            return proc.wait()
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
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except Refused as e:
        say("inkling: not done.")
        for r in e.reasons:
            say(f"  - {r}" if not r.startswith("  ") and not r.endswith(":") else f"  {r}")
        return EXIT_REFUSED
    except Failed as e:
        say(f"inkling: failed: {e}")
        return 1
    except KeyboardInterrupt:
        say("inkling: interrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
