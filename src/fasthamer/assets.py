"""Model asset management + first-run setup.

The CoreML model has MANO mesh data baked into its weights. MANO is
license-gated (https://mano.is.tue.mpg.de), so fasthamer never bundles it in
the wheel. Instead, on first use (or via the `fasthamer-setup` command):

  1. You are asked for your MANO account credentials, and MANO v1.2 is
     downloaded from the official MPI server — registering there is how you
     accept the MANO license. The raw MANO_RIGHT.pkl is kept in the cache
     (handy for driving smplx/MANO with fasthamer's predicted parameters).
  2. The prebuilt CoreML bundle is then downloaded into the cache.

Non-interactive environments can set MANO_USERNAME / MANO_PASSWORD instead.

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
import urllib.error
import urllib.parse
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

# Official MANO download. Unlike the SMPL-X download server, the MANO site
# does not accept POSTed credentials on download.is.tue.mpg.de — the working
# flow is a website login (session cookie) followed by the same-origin
# download/dl.php fetch. Requires an account at https://mano.is.tue.mpg.de.
MANO_LOGIN_URL = "https://mano.is.tue.mpg.de/login.php"
MANO_URL = ("https://mano.is.tue.mpg.de/download/dl.php"
            "?domain=mano&resume=1&sfile=mano_v1_2.zip")
MANO_PKL_IN_ZIP = "mano_v1_2/models/MANO_RIGHT.pkl"


def cache_dir() -> str:
    base = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    return os.path.join(base, "fasthamer")


def mano_pkl_path() -> str:
    return os.path.join(cache_dir(), "mano", "MANO_RIGHT.pkl")


def _is_bundle(path: str) -> bool:
    return (os.path.isdir(os.path.join(path, MODEL_NAME))
            and os.path.isfile(os.path.join(path, FACES_NAME)))


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: str, label: str, opener=None) -> None:
    if opener is not None:
        # openers carry their own headers (and cookies)
        open_ctx = opener.open(url)
    else:
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


def _looks_like_error_page(path: str) -> bool:
    """MPI's download.php returns an HTML page (200) on bad credentials or a
    not-accepted license, instead of an HTTP error."""
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return True
    with open(path, "rb") as f:
        head = f.read(256).lower()
    return (b"<!doctype html" in head or b"<html" in head
            or b"error: file not found." in head)


def download_mano(username: str, password: str) -> str:
    """Log in on the official MANO website with the user's own credentials,
    download MANO v1.2, and stash MANO_RIGHT.pkl in the cache. Returns the
    pkl path."""
    import http.cookiejar
    dest = mano_pkl_path()
    if os.path.isfile(dest):
        return dest
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    # The MANO download endpoint 403s unrecognized User-Agents; the generic
    # browser UA is the one their server accepts.
    opener.addheaders = [("User-Agent", "Mozilla/5.0")]
    creds = urllib.parse.urlencode({"username": username, "password": password,
                                    "commit": "Log in"}).encode()
    os.makedirs(cache_dir(), exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache_dir()) as tmp:
        zip_path = os.path.join(tmp, "mano_v1_2.zip")
        try:
            # GET first to establish the session cookie, then log in with it.
            opener.open(MANO_LOGIN_URL).read()
            opener.open(MANO_LOGIN_URL, data=creds).read()
            _download(MANO_URL, zip_path, "MANO v1.2 (official MPI server)",
                      opener=opener)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise PermissionError(
                    "the MANO server rejected the credentials — check them, "
                    "and that you registered (and accepted the license) at "
                    "https://mano.is.tue.mpg.de") from e
            raise
        if _looks_like_error_page(zip_path) or not zipfile.is_zipfile(zip_path):
            raise PermissionError(
                "the MANO server returned a web page instead of the model — "
                "check your credentials, and that you registered (and "
                "accepted the license) at https://mano.is.tue.mpg.de")
        with zipfile.ZipFile(zip_path) as zf:
            names = [n for n in zf.namelist() if n.endswith("MANO_RIGHT.pkl")]
            if MANO_PKL_IN_ZIP in zf.namelist():
                member = MANO_PKL_IN_ZIP
            elif names:
                member = names[0]
            else:
                raise RuntimeError("MANO_RIGHT.pkl not found in the downloaded "
                                   "mano_v1_2.zip (unexpected archive layout)")
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with zf.open(member) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)
    return dest


def ensure_mano_license(interactive: Optional[bool] = None) -> str:
    """Make sure MANO has been obtained with the user's own MPI account.
    Prompts for credentials on a TTY; honors MANO_USERNAME / MANO_PASSWORD."""
    dest = mano_pkl_path()
    if os.path.isfile(dest):
        return dest

    username = os.environ.get("MANO_USERNAME")
    password = os.environ.get("MANO_PASSWORD")
    if not (username and password):
        if interactive is None:
            interactive = sys.stdin.isatty()
        if not interactive:
            raise RuntimeError(
                "fasthamer needs the license-gated MANO model on first use. "
                "Run `fasthamer-setup` in a terminal, or set the "
                "MANO_USERNAME / MANO_PASSWORD environment variables "
                "(register at https://mano.is.tue.mpg.de)")
        import getpass
        sys.stderr.write(
            "\nfasthamer first-run setup\n"
            "-------------------------\n"
            "The MANO hand model is license-gated. Enter the account you\n"
            "registered at https://mano.is.tue.mpg.de (sign up there first\n"
            "if you have not — it is free for research use).\n\n")
        username = input("Email (MANO account): ").strip()
        password = getpass.getpass("Password (MANO account): ")
    return download_mano(username, password)


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
