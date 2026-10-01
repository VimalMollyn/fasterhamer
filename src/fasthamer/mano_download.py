"""Download mano_v1_2.zip from MPI with the user's own MANO account.

MANO is license-gated: everyone must register at https://mano.is.tue.mpg.de
and accept the license there. This module only automates the download that
the website offers afterwards. Credentials are sent to the MPI servers only
and never stored.

Library use:
    from fasthamer.mano_download import download_mano_zip
    download_mano_zip(username, password, "/path/to/mano_v1_2.zip")

Standalone:
    python -m fasthamer.mano_download [-o OUT] [--extract DIR]
    (or `fasthamer-setup`, which calls this when no local MANO file is found)
"""
import argparse
import getpass
import http.cookiejar
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from typing import Callable, Optional

MANO_SITE = "https://mano.is.tue.mpg.de"
ZIP_NAME = "mano_v1_2.zip"
# Primary: MPI's shared download server; takes username/password as POST form
# data (the same endpoint the SMPL-X / ECON / 4D-Humans fetch scripts use).
DOWNLOAD_URL = ("https://download.is.tue.mpg.de/download.php"
                f"?domain=mano&resume=1&sfile={ZIP_NAME}")
# Fallback: log in on the MANO site for a session cookie, then fetch dl.php.
LOGIN_URL = f"{MANO_SITE}/login.php"
DL_URL = f"{MANO_SITE}/download/dl.php?domain=mano&resume=1&sfile={ZIP_NAME}"
USER_AGENT = "Mozilla/5.0 (compatible; fasthamer)"
USERNAME_ENV = "MANO_USERNAME"
PASSWORD_ENV = "MANO_PASSWORD"

Progress = Optional[Callable[[int, int], None]]  # (bytes_done, bytes_total_or_0)


class ManoDownloadError(RuntimeError):
    pass


class BadCredentials(ManoDownloadError):
    pass


def _opener():
    return urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))


def _is_zip_response(resp) -> bool:
    return resp.status in (200, 206) and "text/html" not in resp.headers.get("Content-Type", "")


def _stream(resp, dest: str, append: bool, progress: Progress) -> None:
    done = os.path.getsize(dest) if append else 0
    total = int(resp.headers.get("Content-Length") or 0) + done
    with open(dest, "ab" if append else "wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if progress:
                progress(done, total)


def _via_download_server(user: str, pw: str, dest: str, resume_from: int,
                         progress: Progress) -> None:
    data = urllib.parse.urlencode({"username": user, "password": pw}).encode()
    headers = {"User-Agent": USER_AGENT}
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"
    req = urllib.request.Request(DOWNLOAD_URL, data=data, headers=headers)
    try:
        resp = _opener().open(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise BadCredentials("username/password rejected by the MPI download server")
        body = e.read(200).decode("utf-8", "replace").strip()
        raise ManoDownloadError(f"HTTP {e.code} from the MPI download server: {body[:120]}")
    with resp:
        if not _is_zip_response(resp):
            raise ManoDownloadError(
                "the MPI download server returned a web page instead of the zip "
                f"(have you accepted the MANO license at {MANO_SITE}?)")
        _stream(resp, dest, append=(resp.status == 206 and resume_from > 0), progress=progress)


def _via_site_login(user: str, pw: str, dest: str, resume_from: int,
                    progress: Progress) -> None:
    opener = _opener()
    form = urllib.parse.urlencode(
        {"username": user, "password": pw, "commit": "Log in"}).encode()
    req = urllib.request.Request(LOGIN_URL, data=form, headers={"User-Agent": USER_AGENT})
    with opener.open(req, timeout=60) as r:
        page = r.read().decode("utf-8", "replace")
    if 'name="password"' in page:  # still on the login form
        raise BadCredentials(f"login at {MANO_SITE} failed (wrong username/password?)")
    headers = {"User-Agent": USER_AGENT}
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"
    with opener.open(urllib.request.Request(DL_URL, headers=headers), timeout=60) as resp:
        if not _is_zip_response(resp):
            raise ManoDownloadError(
                "logged in, but the MANO site returned a web page instead of the zip "
                f"(have you accepted the MANO license at {MANO_SITE}?)")
        _stream(resp, dest, append=(resp.status == 206 and resume_from > 0), progress=progress)


def verify_zip(path: str) -> None:
    with open(path, "rb") as f:
        magic = f.read(2)
    if magic != b"PK":
        raise ManoDownloadError(f"{path} is not a zip file")
    with zipfile.ZipFile(path) as zf:
        bad = zf.testzip()
    if bad is not None:
        raise ManoDownloadError(f"{path} is corrupt (first bad entry: {bad})")


def download_mano_zip(username: str, password: str, dest: str,
                      progress: Progress = None, method: str = "auto") -> str:
    """Download mano_v1_2.zip to `dest` (resumes a partial `<dest>.part`).

    `method`: "auto" tries the MPI download server, then the MANO-site login.
    Raises BadCredentials or ManoDownloadError.
    """
    if not username or not password:
        raise BadCredentials("username and password are required")
    partial = dest + ".part"
    resume_from = os.path.getsize(partial) if os.path.isfile(partial) else 0
    routes = {"auto": (_via_download_server, _via_site_login),
              "download": (_via_download_server,),
              "login": (_via_site_login,)}[method]
    errors = []
    for route in routes:
        try:
            route(username, password, partial, resume_from, progress)
            break
        except BadCredentials:
            raise
        except (ManoDownloadError, urllib.error.URLError, OSError) as e:
            errors.append(f"{route.__name__.lstrip('_')}: {e}")
    else:
        raise ManoDownloadError("could not download MANO:\n  " + "\n  ".join(errors))
    verify_zip(partial)
    os.replace(partial, dest)
    return dest


def prompt_credentials(stream=sys.stderr) -> tuple:
    """Ask for the MANO account email + password (env vars take precedence)."""
    user = os.environ.get(USERNAME_ENV) or input("MANO account email: ").strip()
    pw = os.environ.get(PASSWORD_ENV) or getpass.getpass("MANO password: ")
    return user, pw


def stderr_progress(done: int, total: int) -> None:
    if total:
        sys.stderr.write(f"\r[fasthamer] downloading {ZIP_NAME}... "
                         f"{done / (1 << 20):.0f}/{total / (1 << 20):.0f} MB ({done / total:5.1%})")
    else:
        sys.stderr.write(f"\r[fasthamer] downloading {ZIP_NAME}... {done / (1 << 20):.0f} MB")
    sys.stderr.flush()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m fasthamer.mano_download",
        description="Download mano_v1_2.zip with your MANO account "
                    f"(register + accept the license at {MANO_SITE} first).")
    ap.add_argument("-o", "--out", default=ZIP_NAME,
                    help="output zip path, or a directory to put mano_v1_2.zip in")
    ap.add_argument("--extract", metavar="DIR",
                    help="also extract MANO_RIGHT.pkl / MANO_LEFT.pkl into DIR")
    ap.add_argument("--method", choices=("auto", "download", "login"), default="auto")
    args = ap.parse_args(argv)

    dest = os.path.join(args.out, ZIP_NAME) if os.path.isdir(args.out) else args.out
    user, pw = prompt_credentials()
    try:
        download_mano_zip(user, pw, dest, progress=stderr_progress, method=args.method)
    except ManoDownloadError as e:
        sys.stderr.write(f"\n[fasthamer] {e}\n")
        return 1
    sys.stderr.write(f"\n[fasthamer] saved {dest} ({os.path.getsize(dest) / (1 << 20):.0f} MB)\n")

    if args.extract:
        os.makedirs(args.extract, exist_ok=True)
        with zipfile.ZipFile(dest) as zf:
            for name in zf.namelist():
                base = os.path.basename(name)
                if base in ("MANO_RIGHT.pkl", "MANO_LEFT.pkl"):
                    out = os.path.join(args.extract, base)
                    with zf.open(name) as src, open(out, "wb") as dst:
                        dst.write(src.read())
                    print(f"extracted {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
