"""`fasthamer-setup` — one-time setup.

Confirms you have accepted the MANO license (register at
https://mano.is.tue.mpg.de), then downloads the prebuilt CoreML model bundle
into the fasthamer cache. After this, `fasthamer.load()` works offline.

Non-interactive use:  FASTHAMER_ACCEPT_MANO_LICENSE=1 fasthamer-setup
"""
import argparse
import sys

from .assets import cache_dir, ensure_mano_license, resolve_model_dir


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="fasthamer-setup",
        description="Confirm the MANO license and download the fasthamer "
                    "CoreML model bundle into the cache.")
    ap.parse_args(argv)

    try:
        ensure_mano_license(interactive=sys.stdin.isatty())
        bundle = resolve_model_dir()
    except (PermissionError, RuntimeError) as e:
        print(f"[fasthamer] setup failed: {e}", file=sys.stderr)
        return 1
    print(f"[fasthamer] setup complete.\n"
          f"  cache:      {cache_dir()}\n"
          f"  model:      {bundle}\n"
          f"Try it:  python -c \"import fasthamer; print(fasthamer.load())\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
