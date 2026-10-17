"""Model asset management + first-run setup.

Everything lives in `~/.cache/fasthamer` and is fetched on first use:

  1. **The CoreML model bundle** (ViT-H backbone + MANO head, ~470 MB) —
     downloaded from the GitHub release. It contains no MANO data: it stops
     at the MANO *parameters*.
  2. **The SOMA-X hand assets** (~40 MB, default `hand_model="soma"`): the
     mano2soma-x regressor (CoreML + npz) and NVIDIA's Apache-licensed SOMA-X
     right-hand model data, downloaded from the GitHub release. No account,
     no prompt (see `fasthamer.soma_np`).
  3. **Your own MANO model** (`MANO_RIGHT.pkl`), only for `hand_model="mano"`.
     MANO is license-gated (https://mano.is.tue.mpg.de, free for
     non-commercial research), so fasthamer never ships it. `fasthamer-setup
     --mano` (or the first `load(hand_model="mano")`) either imports a copy
     you already have or downloads `mano_v1_2.zip` with your MANO account
     credentials (sent only to MPI, never stored). The pkl is converted once
     to a small `.npz`; the mesh is then computed in numpy (`fasthamer.mano`).

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

Resolution order for the SOMA-X asset directory (same idea):
  `soma_dir=` argument, FASTHAMER_SOMA_DIR, the cache (first-run download).

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

SOMA_ASSETS_VERSION = 1
SOMA_ASSETS_URL = ("https://github.com/VimalMollyn/fasterhamer/releases/download/"
                   f"soma-v{SOMA_ASSETS_VERSION}/fasthamer-soma-v{SOMA_ASSETS_VERSION}.zip")
SOMA_ASSETS_SHA256: Optional[str] = \
    "cfffbcbb60f61259635dde868fbf13232c7dae3660e394359c3351210ba6783d"
SOMA_NPZ_NAME = "mano2somax_resmlp512x4_numpy.npz"
SOMA_REGRESSOR_NAME = "mano2somax_regressor_fp32.mlpackage"

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


def _fetch_zip(url: str, sha256: Optional[str], dest: str, is_valid, label: str,
               hint: str) -> None:
    """Download a zip, verify it, extract it and move the directory that satisfies
    `is_valid` (the zip root or a single top-level folder) to `dest`."""
    os.makedirs(cache_dir(), exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache_dir()) as tmp:
        zip_path = os.path.join(tmp, "assets.zip")
        try:
            _download(url, zip_path, label)
        except Exception as e:
            raise RuntimeError(f"failed to download the fasthamer {label} from {url} — "
                               f"check your connection, or {hint}") from e
        if sha256 is not None:
            digest = _sha256(zip_path)
            if digest != sha256:
                raise RuntimeError(f"{label} checksum mismatch (got {digest}, expected "
                                   f"{sha256}) — the download may be corrupt")
        extract_dir = os.path.join(tmp, "extracted")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
        root = extract_dir
        if not is_valid(root):
            entries = [os.path.join(root, e) for e in os.listdir(root)]
            subdirs = [e for e in entries if os.path.isdir(e) and is_valid(e)]
            if not subdirs:
                raise RuntimeError(f"downloaded {label} from {url} has an unexpected "
                                   f"layout: {os.listdir(root)}")
            root = subdirs[0]
        shutil.move(root, dest)


def _fetch_bundle(dest: str) -> None:
    """Download + extract the prebuilt CoreML bundle into `dest`."""
    url = os.environ.get("FASTHAMER_ASSETS_URL", ASSETS_URL)
    _fetch_zip(url, ASSETS_SHA256 if url == ASSETS_URL else None, dest, _is_bundle,
               "CoreML model bundle",
               "set FASTHAMER_MODEL_DIR / pass model_dir= to point at a local bundle")


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
# SOMA-X hand assets (default hand model; no license gate)
# ----------------------------------------------------------------------------
def soma_paths(soma_dir: str) -> Tuple[str, Optional[str]]:
    """(npz path, CoreML regressor path or None) inside a SOMA asset directory."""
    npz = os.path.join(soma_dir, SOMA_NPZ_NAME)
    reg = os.path.join(soma_dir, SOMA_REGRESSOR_NAME)
    return npz, (reg if os.path.isdir(reg) else None)


def _is_soma_dir(path: str) -> bool:
    return os.path.isfile(os.path.join(path, SOMA_NPZ_NAME))


def resolve_soma_dir(soma_dir: Optional[str] = None, download: bool = True) -> str:
    """Return the directory holding the SOMA-X assets, downloading them into the
    cache on first use."""
    for cand in (soma_dir, os.environ.get("FASTHAMER_SOMA_DIR")):
        if cand:
            cand = os.path.expanduser(cand)
            if _is_soma_dir(cand):
                return cand
            raise FileNotFoundError(f"'{cand}' is not a fasthamer SOMA asset directory "
                                    f"(expected {SOMA_NPZ_NAME} inside it)")
    cached = os.path.join(cache_dir(), f"soma-v{SOMA_ASSETS_VERSION}")
    if _is_soma_dir(cached):
        return cached
    if not download:
        raise FileNotFoundError(f"SOMA assets not found at {cached}")
    url = os.environ.get("FASTHAMER_SOMA_ASSETS_URL", SOMA_ASSETS_URL)
    _fetch_zip(url, SOMA_ASSETS_SHA256 if url == SOMA_ASSETS_URL else None, cached,
               _is_soma_dir, "SOMA-X hand assets",
               "set FASTHAMER_SOMA_DIR to a local copy, or use hand_model='mano'")
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
