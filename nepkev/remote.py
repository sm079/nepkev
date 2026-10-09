"""Run train_eval.sh (next to this module) on one RunPod GPU pod: launch, upload, run, download, always terminate.

The pod is SSH-only (no public HTTP ports), uses the account's SSH key, and is terminated in a `finally` block and by
a watchdog thread at a hard deadline, so a hung connection cannot leave it billing. Needs RUNPOD_API_KEY.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel

from . import CONFIGS, ROOT

REST = "https://rest.runpod.io/v1"
SCRIPT = Path(__file__).with_name("train_eval.sh")


class RemoteConfig(BaseModel):
    gpu_types: list[str]
    cloud_type: str
    image: str
    allowed_cuda_versions: list[str] = []
    container_disk_gb: int
    max_cost_per_hr: float
    max_minutes: int
    kev_commit: str
    kev_model: str
    kev_revision: str
    kev_base: str
    kev_base_revision: str
    lr: float
    batch: int
    accum: int
    epochs: int = 2          # each one evaluated on dev; the lowest dev NLL is selected
    seed: int


def load_remote_config(name: str) -> RemoteConfig:
    p = Path(name) if Path(name).is_file() else CONFIGS / "remote" / f"{name}.yaml"   # "kev-0.8b" has a "suffix" .8b
    cfg = RemoteConfig.model_validate(yaml.safe_load(p.read_text(encoding="utf-8")))
    up = yaml.safe_load((CONFIGS / "upstream.yaml").read_text(encoding="utf-8"))
    m = next((m for m in up["models"].values() if m["hub_id"] == cfg.kev_model), None)
    if m is None:
        raise ValueError(f"{p}: {cfg.kev_model} is not a model pinned in configs/upstream.yaml")
    if (cfg.kev_commit, cfg.kev_revision, cfg.kev_base, cfg.kev_base_revision) != (up["commit"], m["revision"], m["base"],
                                                                                   m["base_revision"]):
        raise ValueError(f"{p} pins differ from configs/upstream.yaml")
    return cfg


def _api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key and sys.platform == "win32":
        import winreg   # a user variable set after this shell started is only in the registry
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                key = winreg.QueryValueEx(k, "RUNPOD_API_KEY")[0]
        except OSError:
            pass
    if not key:
        raise RuntimeError("RUNPOD_API_KEY is not set")
    return key


def api(method: str, path: str, body: dict | None = None) -> dict | None:
    req = urllib.request.Request(REST + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json",
                                          "User-Agent": "nepkev-remote/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"RunPod {method} {path}: {e.code} {e.read().decode(errors='replace')}") from e
    return json.loads(raw) if raw else None


@dataclass
class Pod:
    id: str
    ip: str = ""
    port: int = 0
    cost_per_hr: float | None = None


def _ssh(pod: Pod, *extra: str) -> list[str]:
    known = ROOT / "data" / "remote" / f"known_hosts_{pod.id}"
    known.parent.mkdir(parents=True, exist_ok=True)
    return ["ssh", "-i", os.path.expanduser("~/.ssh/id_ed25519"), "-p", str(pod.port), "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"UserKnownHostsFile={known}", "-o", "ServerAliveInterval=30", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes",
            *extra, f"root@{pod.ip}"]


def launch(cfg: RemoteConfig, name: str) -> Pod:
    body = {"name": name, "imageName": cfg.image, "gpuTypeIds": cfg.gpu_types, "gpuCount": 1, "cloudType": cfg.cloud_type,
            "containerDiskInGb": cfg.container_disk_gb, "volumeInGb": 0, "ports": ["22/tcp"], "supportPublicIp": True}
    if cfg.allowed_cuda_versions:
        body["allowedCudaVersions"] = cfg.allowed_cuda_versions
    created = api("POST", "/pods", body)
    pod = Pod(created["id"], cost_per_hr=created.get("costPerHr"))
    print(f"pod {pod.id} created: {created.get('machine', {}).get('gpuDisplayName') or created.get('gpu', '')} "
          f"${pod.cost_per_hr}/hr", flush=True)
    return pod


def wait_for_ssh(pod: Pod, timeout_s: int = 900) -> Pod:
    start = time.time()
    while time.time() - start < timeout_s:
        info = api("GET", f"/pods/{pod.id}") or {}
        ip, port = info.get("publicIp"), (info.get("portMappings") or {}).get("22")
        if ip and port:
            pod.ip, pod.port = ip, int(port)
            if subprocess.run(_ssh(pod) + ["true"], capture_output=True).returncode == 0:
                print(f"ssh ready after {int(time.time() - start)}s", flush=True)
                return pod
        time.sleep(10)
    raise TimeoutError(f"pod {pod.id}: no SSH after {timeout_s}s")


def terminate(pod: Pod) -> None:
    try:
        api("DELETE", f"/pods/{pod.id}")
        print(f"pod {pod.id} terminated", flush=True)
    except RuntimeError as e:
        print(f"TERMINATE FAILED for pod {pod.id}: {e} -- terminate it in the RunPod console", flush=True)


def _job_tar(kev_dir: Path, cfg: RemoteConfig) -> bytes:
    env = {"KEV_COMMIT": cfg.kev_commit, "KEV_MODEL": cfg.kev_model, "KEV_REVISION": cfg.kev_revision, "KEV_BASE": cfg.kev_base,
           "KEV_BASE_REVISION": cfg.kev_base_revision, "LR": cfg.lr, "BATCH": cfg.batch, "ACCUM": cfg.accum, "EPOCHS": cfg.epochs, "SEED": cfg.seed}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        def add_bytes(name: str, data: bytes) -> None:
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o755
            tar.addfile(info, io.BytesIO(data))
        add_bytes("job.env", "".join(f"{k}={v}\n" for k, v in env.items()).encode())
        add_bytes("train_eval.sh", SCRIPT.read_bytes().replace(b"\r\n", b"\n"))
        for f in sorted(kev_dir.glob("*.jsonl")):
            add_bytes(f"kev/{f.name}", f.read_bytes())
    return buf.getvalue()


def download(pod, out_dir: Path) -> bool:
    """Copy log.txt and the whole runs directory (evaluations, selection, calibration, checkpoints) off the pod."""
    if not pod.ip:
        return False
    try:
        dl = subprocess.run(_ssh(pod) + ["cd /workspace/job && tar czf - log.txt runs --exclude='*.safetensors.tmp' 2>/dev/null"],
                            capture_output=True, timeout=1800)
    except subprocess.TimeoutExpired:
        print("download timed out", flush=True)
        return False
    if not dl.stdout:
        return False
    with tarfile.open(fileobj=io.BytesIO(dl.stdout), mode="r:gz") as tar:
        tar.extractall(out_dir, filter="data")
    print(f"results downloaded to {out_dir}", flush=True)
    return True


def run_job(kev_dir: Path, out_dir: Path, cfg: RemoteConfig, name: str) -> int:
    """Launch, run, download, terminate. Returns the remote exit code (non-zero: see out_dir/log.txt).
    Results are downloaded before every termination, including the watchdog's at the deadline."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pod = launch(cfg, name)
    deadline = time.time() + (cfg.max_minutes + 10) * 60
    done = threading.Event()

    def watchdog() -> None:
        while not done.wait(30):
            if time.time() > deadline:
                print("deadline reached: saving results, then terminating the pod", flush=True)
                try:
                    download(pod, out_dir)
                finally:
                    terminate(pod)
                return
    threading.Thread(target=watchdog, daemon=True).start()
    rc = -1
    try:
        if pod.cost_per_hr is not None and pod.cost_per_hr > cfg.max_cost_per_hr:
            raise RuntimeError(f"pod costs ${pod.cost_per_hr}/hr > max_cost_per_hr {cfg.max_cost_per_hr}")
        wait_for_ssh(pod)
        up = subprocess.run(_ssh(pod) + ["mkdir -p /workspace/job && tar xzf - -C /workspace/job"], input=_job_tar(kev_dir, cfg))
        if up.returncode:
            raise RuntimeError("upload failed")
        cmd = f"cd /workspace/job && timeout {cfg.max_minutes}m bash train_eval.sh 2>&1 | tee /workspace/job/log.txt; exit ${{PIPESTATUS[0]}}"
        rc = subprocess.run(_ssh(pod) + [cmd]).returncode
        print(f"remote job exited with {rc}", flush=True)
    finally:
        try:
            download(pod, out_dir)
        finally:
            done.set()
            terminate(pod)
    return rc
