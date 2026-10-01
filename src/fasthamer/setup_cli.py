"""`fasthamer-setup` — one-time setup.

  1. MANO: imports your MANO model (MANO_RIGHT.pkl / mano_v1_2 folder / zip)
     into the fasthamer cache, or downloads mano_v1_2.zip with your MANO
     account (register + accept the license at https://mano.is.tue.mpg.de).
  2. Downloads the prebuilt CoreML model bundle (~470 MB) into the cache.

After this, `fasthamer.load()` works offline.

    fasthamer-setup                         # interactive
    fasthamer-setup --mano ~/mano_v1_2.zip  # use a MANO file you already have
    MANO_USERNAME=... MANO_PASSWORD=... fasthamer-setup   # non-interactive
"""
import argparse
import sys

from .assets import cache_dir, resolve_mano, resolve_model_dir


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="fasthamer-setup",
        description="Set up fasthamer: import/download your MANO model and "
                    "download the CoreML model bundle into the cache.")
    ap.add_argument("--mano", metavar="PATH",
                    help="MANO_RIGHT.pkl, the mano_v1_2 folder, or mano_v1_2.zip "
                         "(default: cached copy, auto-detect, or download with your "
                         "MANO account)")
    args = ap.parse_args(argv)

    try:
        mano = resolve_mano(args.mano, interactive=sys.stdin.isatty())
        bundle = resolve_model_dir()
    except (PermissionError, RuntimeError, FileNotFoundError, ValueError) as e:
        print(f"[fasthamer] setup failed: {e}", file=sys.stderr)
        return 1
    print(f"[fasthamer] setup complete.\n"
          f"  cache:      {cache_dir()}\n"
          f"  MANO:       {mano}\n"
          f"  model:      {bundle}\n"
          f"Try it live:  fasthamer-webcam --mirror\n"
          f"Or in Python: python -c \"import fasthamer; print(fasthamer.load())\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
