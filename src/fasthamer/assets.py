"""Model asset management + first-run setup.

fasthamer needs two things on disk, both kept in `~/.cache/fasthamer`:

  1. **The CoreML model bundle** (ViT-H backbone + MANO head, ~470 MB) —
     downloaded from the GitHub release on first use. It contains no MANO
     data: it stops at the MANO *parameters*.
  2. **Your own MANO model** (`MANO_RIGHT.pkl`). MANO is license-gated
     (https://mano.is.tue.mpg.de, free for non-commercial research), so
     fasthamer never ships it. On first use (or via `fasthamer-setup`) you
     either point fasthamer at a copy you already have, or let it download
     `mano_v1_2.zip` with your MANO account credentials (sent only to MPI,
     never stored). The pkl is converted once to a small `.npz`; the mesh is
     then computed in numpy at runtime (see `fasthamer.mano`).

Resolution order for the MANO model:
  1. `mano_path=` argument to `fasthamer.load()` / `HandMesh()` /
     `fasthamer-setup --mano` — a `MANO_RIGHT.pkl`, a folder containing it
     (e.g. `mano_v1_2/` or `mano_v1_2/models/`), or `mano_v1_2.zip`
  2. the FASTHAMER_MANO_PATH environment variable (same forms)
  3. the fasthamer cache (`~/.cache/fasthamer/mano/mano_right.npz`)
  4. auto-detection in a few conventional places (`./_DATA/data/mano/`,
     `./MANO_RIGHT.pkl`, `~/Downloads/mano_v1_2*`)
  5. download with MANO_USERNAME / MANO_PASSWORD (env, non-interactive) or an
     interactive prompt (path, or email + password)

Resolution order for the model bundle directory:
  1. `model_dir=` argument to `fasthamer.load()` / `HandMesh()`
  2. the FASTHAMER_MODEL_DIR environment variable
  3. the fasthamer cache, populated by the first-run download

A current bundle directory contains `hamer_params.mlpackage`. Legacy bundles
(`hamer_mano.mlpackage` + `mano_faces.npy`, mesh baked in) still load and
need no MANO file.
"""
import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from typing import Optional, Tuple

from .mano import (MANO_NPZ_NAME, PKL_NAME, extract_right_pkl, load_mano_npz,
                   load_mano_pkl, save_mano_npz)

ASSETS_VERSION = 2
ASSETS_URL = ("https://github.com/VimalMollyn/fasterhamer/releases/download/"
              f"assets-v{ASSETS_VERSION}/fasthamer-assets-v{ASSETS_VERSION}.zip")
# sha256 of the assets zip; update alongside ASSETS_URL when publishing a bundle.
ASSETS_SHA256: Optional[str] = \
    "5ff4745a5a0cc48b1bd18e55867f177cccb982d1dc1dd1d6b06090c3e0a5e7d8"

MODEL_NAME = "hamer_params.mlpackage"        # backbone + MANO head (no MANO data)
LEGACY_MODEL_NAME = "hamer_mano.mlpackage"   # v1 bundles: MANO mesh baked in
FACES_NAME = "mano_faces.npy"                # legacy bundles only

MANO_URL = "https://mano.is.tue.mpg.de"
MANO_PATH_ENV = "FASTHAMER_MANO_PATH"


def cache_dir() -> str:
    base = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    return os.path.join(base, "fasthamer")


def mano_cache_dir() -> str:
    return os.path.join(cache_dir(), "mano")


def mano_npz_path() -> str:
    return os.path.join(mano_cache_dir(), MANO_NPZ_NAME)


# ----------------------------------------------------------------------------
# model bundle
# ----------------------------------------------------------------------------
def bundle_model_path(bundle_dir: str) -> Tuple[str, bool]:
    """(path to the mlpackage inside `bundle_dir`, is_legacy_fused_bundle)."""
    new = os.path.join(bundle_dir, MODEL_NAME)
    if os.path.isdir(new):
        return new, False
    old = os.path.join(bundle_dir, LEGACY_MODEL_NAME)
    if os.path.isdir(old) and os.path.isfile(os.path.join(bundle_dir, FACES_NAME)):
        return old, True
    raise FileNotFoundError(f"'{bundle_dir}' is not a fasthamer model bundle "
                            f"(expected {MODEL_NAME}/ inside it)")


def _is_bundle(path: str) -> bool:
    try:
        bundle_model_path(path)
        return True
    except FileNotFoundError:
        return False


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
        if ASSETS_SHA256 is not None and url == ASSETS_URL:
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
    """Return a directory containing the CoreML model bundle, downloading it
    into the cache on first use."""
    for cand in (model_dir, os.environ.get("FASTHAMER_MODEL_DIR")):
        if cand:
            cand = os.path.expanduser(cand)
            bundle_model_path(cand)  # raises a descriptive FileNotFoundError
            return cand

    cached = os.path.join(cache_dir(), f"assets-v{ASSETS_VERSION}")
    if _is_bundle(cached):
        return cached
    if not download:
        raise FileNotFoundError(f"model bundle not found at {cached}")
    _fetch_bundle(cached)
    return cached


# ----------------------------------------------------------------------------
# MANO model
# ----------------------------------------------------------------------------
def _find_pkl_in_dir(d: str) -> Optional[str]:
    for sub in ("", "models", os.path.join("mano_v1_2", "models"), "mano"):
        p = os.path.join(d, sub, PKL_NAME)
        if os.path.isfile(p):
            return p
    return None


def import_mano(source: str) -> str:
    """Import a MANO model from `source` into the fasthamer cache and return
    the cached .npz path. `source` may be MANO_RIGHT.pkl, a directory that
    contains it (`mano_v1_2/`, `mano_v1_2/models/`, ...), or mano_v1_2.zip."""
    src = os.path.expanduser(source)
    cached_pkl = os.path.join(mano_cache_dir(), PKL_NAME)
    if os.path.isdir(src):
        pkl = _find_pkl_in_dir(src)
        if pkl is None:
            raise FileNotFoundError(f"no {PKL_NAME} found under {src}")
    elif os.path.isfile(src) and src.lower().endswith(".zip"):
        pkl = extract_right_pkl(src, cached_pkl)
    elif os.path.isfile(src):
        pkl = src
    else:
        raise FileNotFoundError(f"MANO model not found: {src}")

    data = load_mano_pkl(pkl)  # validates it really is MANO_RIGHT
    os.makedirs(mano_cache_dir(), exist_ok=True)
    if os.path.abspath(pkl) != os.path.abspath(cached_pkl):
        shutil.copyfile(pkl, cached_pkl)
    save_mano_npz(data, mano_npz_path())
    return mano_npz_path()


def _auto_detect_mano() -> Optional[str]:
    home = os.path.expanduser("~")
    candidates = [
        os.path.join("_DATA", "data", "mano", PKL_NAME),  # HaMeR repo layout
        PKL_NAME,
        os.path.join("mano", PKL_NAME),
        os.path.join("mano_v1_2", "models", PKL_NAME),
        "mano_v1_2.zip",
        os.path.join(home, "Downloads", "mano_v1_2", "models", PKL_NAME),
        os.path.join(home, "Downloads", "mano_v1_2.zip"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def _download_mano(username: str, password: str) -> str:
    """Download mano_v1_2.zip with the user's credentials, keep MANO_RIGHT.pkl
    in the cache, build the .npz, drop the zip."""
    from .mano_download import download_mano_zip, stderr_progress
    os.makedirs(mano_cache_dir(), exist_ok=True)
    zip_path = os.path.join(mano_cache_dir(), "mano_v1_2.zip")
    download_mano_zip(username, password, zip_path, progress=stderr_progress)
    sys.stderr.write("\n")
    try:
        return import_mano(zip_path)
    finally:
        try:
            os.remove(zip_path)
        except OSError:
            pass


def _obtain_mano(interactive: Optional[bool]) -> str:
    from .mano_download import (PASSWORD_ENV, USERNAME_ENV, BadCredentials,
                                ManoDownloadError, prompt_credentials)
    user, pw = os.environ.get(USERNAME_ENV), os.environ.get(PASSWORD_ENV)
    if user and pw:
        sys.stderr.write(f"[fasthamer] downloading MANO with {USERNAME_ENV} credentials\n")
        return _download_mano(user, pw)

    if interactive is None:
        interactive = sys.stdin.isatty()
    if not interactive:
        raise RuntimeError(
            "fasthamer needs the MANO hand model (MANO_RIGHT.pkl), which is "
            f"license-gated and must come from your own account at {MANO_URL}. "
            f"Run `fasthamer-setup` in a terminal, or set {MANO_PATH_ENV} to your "
            f"MANO_RIGHT.pkl / mano_v1_2.zip, or set {USERNAME_ENV} and "
            f"{PASSWORD_ENV} to let fasthamer download it.")

    sys.stderr.write(
        "\nfasthamer first-run setup: MANO hand model\n"
        "------------------------------------------\n"
        "fasthamer computes the hand mesh with MANO, which is license-gated\n"
        "(free for non-commercial research) and must come from your own MANO\n"
        f"account: {MANO_URL}  (register + accept the license there)\n\n"
        "  * If you already have it: enter the path to MANO_RIGHT.pkl, to the\n"
        "    mano_v1_2 folder, or to mano_v1_2.zip.\n"
        "  * Otherwise press Enter to download it now with your MANO account\n"
        "    email + password (sent only to the MPI server, never stored).\n\n")
    for attempt in range(3):
        reply = input("Path to MANO (or Enter to download): ").strip()
        if reply:
            try:
                return import_mano(reply)
            except (FileNotFoundError, ValueError) as e:
                sys.stderr.write(f"[fasthamer] {e}\n")
                continue
        user, pw = prompt_credentials()
        try:
            return _download_mano(user, pw)
        except BadCredentials as e:
            sys.stderr.write(f"[fasthamer] {e}\n")
        except ManoDownloadError as e:
            raise RuntimeError(
                f"{e}\nMake sure you have registered AND accepted the license at "
                f"{MANO_URL} (the download page there must work in your browser), "
                "or download mano_v1_2.zip in the browser and re-run with "
                f"`fasthamer-setup --mano /path/to/mano_v1_2.zip`.") from None
    raise RuntimeError("could not obtain the MANO model (too many failed attempts)")


def resolve_mano(mano_path: Optional[str] = None, download: bool = True,
                 interactive: Optional[bool] = None) -> str:
    """Return the path to the cached MANO .npz, importing / downloading the
    user's MANO model first if needed (see the module docstring for the
    resolution order)."""
    for cand in (mano_path, os.environ.get(MANO_PATH_ENV)):
        if cand:
            cand = os.path.expanduser(cand)
            if cand.lower().endswith(".npz") and os.path.isfile(cand):
                load_mano_npz(cand)  # validate
                return cand
            return import_mano(cand)

    npz = mano_npz_path()
    if os.path.isfile(npz):
        return npz
    found = _auto_detect_mano()
    if found is not None:
        sys.stderr.write(f"[fasthamer] using the MANO model found at {found}\n")
        return import_mano(found)
    if not download:
        raise FileNotFoundError(f"MANO model not found at {npz}")
    return _obtain_mano(interactive)
