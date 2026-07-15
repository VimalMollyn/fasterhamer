"""`fasthamer-setup` — one-time interactive setup.

Asks for your MANO account credentials (register at https://mano.is.tue.mpg.de),
downloads MANO v1.2 from the official MPI server, then fetches the prebuilt
CoreML model bundle into the fasthamer cache. After this, `fasthamer.load()`
works offline.

Non-interactive use:  MANO_USERNAME=... MANO_PASSWORD=... fasthamer-setup
"""
import argparse
import sys

from .assets import (cache_dir, ensure_mano_license, mano_pkl_path,
                     resolve_model_dir)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="fasthamer-setup",
        description="Download the MANO model (with your own MPI credentials) "
                    "and the fasthamer CoreML bundle into the cache.")
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
          f"  MANO pkl:   {mano_pkl_path()}\n"
          f"Try it:  python -c \"import fasthamer; print(fasthamer.load())\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
