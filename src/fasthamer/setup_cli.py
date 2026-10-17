"""`fasthamer-setup` — optional one-time setup (everything also happens lazily
on the first `fasthamer.load()`).

  * Downloads the CoreML model bundle (~470 MB) and the SOMA-X hand assets
    (~40 MB) into the cache, so later loads work offline. No account needed.
  * `--mano [PATH]`: additionally set up the MANO hand model for
    `hand_model="mano"` — import a MANO_RIGHT.pkl / mano_v1_2 folder / zip you
    already have, or download mano_v1_2.zip with your MANO account
    (register + accept the license at https://mano.is.tue.mpg.de).

    fasthamer-setup                          # SOMA (default hand model) + model bundle
    fasthamer-setup --mano                   # ... and MANO, prompting for a path or your account
    fasthamer-setup --mano ~/mano_v1_2.zip   # ... and MANO from a file you have
    MANO_USERNAME=... MANO_PASSWORD=... fasthamer-setup --mano   # non-interactive
"""
import argparse
import sys

from .assets import cache_dir, resolve_mano, resolve_model_dir, resolve_soma_dir


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="fasthamer-setup",
        description="Download fasthamer's model bundle and SOMA-X hand assets into the "
                    "cache; optionally set up your MANO model too.")
    ap.add_argument("--mano", nargs="?", const="", default=None, metavar="PATH",
                    help="also set up MANO (hand_model='mano'): PATH is MANO_RIGHT.pkl, the "
                         "mano_v1_2 folder, or mano_v1_2.zip; without PATH fasthamer looks for "
                         "a cached/auto-detected copy or asks for a path or your MANO account")
    ap.add_argument("--no-soma", action="store_true", help="skip the SOMA-X assets")
    args = ap.parse_args(argv)

    lines = []
    try:
        bundle = resolve_model_dir()
        lines.append(f"  model:      {bundle}")
        if not args.no_soma:
            lines.append(f"  SOMA-X:     {resolve_soma_dir()}")
        if args.mano is not None:
            mano = resolve_mano(args.mano or None, interactive=sys.stdin.isatty())
            lines.append(f"  MANO:       {mano}")
    except (PermissionError, RuntimeError, FileNotFoundError, ValueError) as e:
        print(f"[fasthamer] setup failed: {e}", file=sys.stderr)
        return 1
    print("[fasthamer] setup complete.\n"
          f"  cache:      {cache_dir()}\n" + "\n".join(lines) + "\n"
          "Try it live:  fasthamer-webcam --mirror\n"
          "Or in Python: python -c \"import fasthamer; print(fasthamer.load())\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
