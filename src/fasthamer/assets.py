"""Model asset management + first-run setup.

The CoreML model has MANO mesh data baked into its weights. MANO is
license-gated (https://mano.is.tue.mpg.de), so fasthamer never bundles it in
the wheel. Instead, on first use (or via the `fasthamer-setup` command):

  1. You confirm (once) that you have registered at https://mano.is.tue.mpg.de
     and accepted the MANO license — a quick acknowledgment, no large download.
  2. The prebuilt CoreML bundle is then downloaded into the cache.

Non-interactive environments accept the license by setting
FASTHAMER_ACCEPT_MANO_LICENSE=1.

Resolution order for the model bundle directory:
  1. `model_dir=` argument to `fasthamer.load()` / `HandMesh()`
  2. the FASTHAMER_MODEL_DIR environment variable (skips the license step —
     use it if you already have a locally built bundle)
  3. the fasthamer cache, populated by the first-run setup above

A valid bundle directory contains `hamer_mano.mlpackage` and `mano_faces.npy`.
"""
import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from typing import Optional

ASSETS_VERSION = 1
ASSETS_URL = ("https://github.com/VimalMollyn/fasterhamer/releases/download/"
              f"assets-v{ASSETS_VERSION}/fasthamer-assets-v{ASSETS_VERSION}.zip")
# sha256 of the assets zip; update alongside ASSETS_URL when publishing a bundle.
ASSETS_SHA256: Optional[str] = \
    "8fa5c3d035854bfdf29732ae4c126f34b3927310e331493674509c9e6ab18113"

MODEL_NAME = "hamer_mano.mlpackage"
FACES_NAME = "mano_faces.npy"

MANO_URL = "https://mano.is.tue.mpg.de"
# Env var to accept the MANO license non-interactively (CI, scripts).
MANO_ACCEPT_ENV = "FASTHAMER_ACCEPT_MANO_LICENSE"


def cache_dir() -> str:
    base = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    return os.path.join(base, "fasthamer")


def _license_marker() -> str:
    return os.path.join(cache_dir(), "mano_license_accepted")


def _is_bundle(path: str) -> bool:
    return (os.path.isdir(os.path.join(path, MODEL_NAME))
            and os.path.isfile(os.path.join(path, FACES_NAME)))


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: str, label: str) -> None:
    open_ctx = urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": "fasthamer"}))
    with open_ctx as resp, open(dest, "wb") as f:
        total = int(resp.headers.get("Content-Length") or 0)
        got = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if total:
                sys.stderr.write(f"\r[fasthamer] downloading {label}... "
                                 f"{got / total:5.1%}")
            else:
                sys.stderr.write(f"\r[fasthamer] downloading {label}... "
                                 f"{got / (1 << 20):.0f} MB")
            sys.stderr.flush()
    sys.stderr.write("\n")


def ensure_mano_license(interactive: Optional[bool] = None) -> None:
    """Make sure the user has acknowledged the MANO license before the
    MANO-derived model bundle is downloaded. The acknowledgment is recorded in
    the cache so it is only asked once. Honors FASTHAMER_ACCEPT_MANO_LICENSE=1
    for non-interactive use."""
    if os.path.isfile(_license_marker()):
        return
    if os.environ.get(MANO_ACCEPT_ENV, "").strip().lower() in ("1", "true", "yes"):
        _record_license_acceptance()
        return

    if interactive is None:
        interactive = sys.stdin.isatty()
    if not interactive:
        raise RuntimeError(
            "fasthamer is built on the license-gated MANO hand model. "
            "Register and accept the license at https://mano.is.tue.mpg.de, "
            f"then set {MANO_ACCEPT_ENV}=1 (or run `fasthamer-setup` in a "
            "terminal) to confirm.")

    sys.stderr.write(
        "\nfasthamer first-run setup\n"
        "-------------------------\n"
        "fasthamer is built on the MANO hand model, which is free for\n"
        "non-commercial research but license-gated. Before downloading the\n"
        "model, please confirm that you have registered at\n"
        "https://mano.is.tue.mpg.de and accepted the MANO license.\n\n")
    reply = input("Have you registered and accepted the MANO license? [y/N]: ")
    if reply.strip().lower() not in ("y", "yes"):
        raise RuntimeError(
            "MANO license not confirmed. Register and accept it at "
            "https://mano.is.tue.mpg.de, then re-run.")
    _record_license_acceptance()


def _record_license_acceptance() -> None:
    os.makedirs(cache_dir(), exist_ok=True)
    with open(_license_marker(), "w") as f:
        f.write("MANO license acknowledged via fasthamer "
                "(https://mano.is.tue.mpg.de)\n")


def _fetch_bundle(dest: str) -> None:
    """Download + extract the prebuilt CoreML bundle into `dest`."""
    url = os.environ.get("FASTHAMER_ASSETS_URL", ASSETS_URL)
    os.makedirs(cache_dir(), exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache_dir()) as tmp:
        zip_path = os.path.join(tmp, "assets.zip")
        try:
            _download(url, zip_path, "CoreML model bundle")
        except Exception as e:
            raise RuntimeError(
                f"failed to download the fasthamer model bundle from {url} — "
                "check your connection, or set FASTHAMER_MODEL_DIR / pass "
                "model_dir= to point at a local bundle") from e
        if ASSETS_SHA256 is not None:
            digest = _sha256(zip_path)
            if digest != ASSETS_SHA256:
                raise RuntimeError(
                    f"model bundle checksum mismatch (got {digest}, "
                    f"expected {ASSETS_SHA256}) — the download may be corrupt")
        extract_dir = os.path.join(tmp, "extracted")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
        # Accept both a flat zip and one with a single top-level directory.
        root = extract_dir
        if not _is_bundle(root):
            entries = [os.path.join(root, e) for e in os.listdir(root)]
            subdirs = [e for e in entries if os.path.isdir(e) and _is_bundle(e)]
            if not subdirs:
                raise RuntimeError(f"downloaded bundle from {url} has an "
                                   f"unexpected layout: {os.listdir(root)}")
            root = subdirs[0]
        shutil.move(root, dest)


def resolve_model_dir(model_dir: Optional[str] = None, download: bool = True,
                      interactive: Optional[bool] = None) -> str:
    """Return a directory containing the CoreML model + MANO faces, running the
    first-run setup (MANO license check + bundle download) if needed."""
    for cand in (model_dir, os.environ.get("FASTHAMER_MODEL_DIR")):
        if cand:
            cand = os.path.expanduser(cand)
            if _is_bundle(cand):
                return cand
            raise FileNotFoundError(
                f"'{cand}' is not a fasthamer model bundle "
                f"(expected {MODEL_NAME}/ and {FACES_NAME} inside it)")

    cached = os.path.join(cache_dir(), f"assets-v{ASSETS_VERSION}")
    if _is_bundle(cached):
        return cached
    if not download:
        raise FileNotFoundError(f"model bundle not found at {cached}")
    ensure_mano_license(interactive)
    _fetch_bundle(cached)
    return cached
