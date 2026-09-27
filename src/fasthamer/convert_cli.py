"""`fasthamer-convert-torch` — build the PyTorch model bundle from the official HaMeR checkpoint.

    fasthamer-convert-torch --ckpt /path/to/hamer.ckpt            # -> ~/.cache/fasthamer/torch-v1
    fasthamer-convert-torch --ckpt hamer.ckpt --out ./bundle --dtype float32

The bundle (hamer_torch.pt + mano_faces.npy) is what `fasthamer.load(backend="torch")` loads.
Get hamer.ckpt from the HaMeR release (https://github.com/geopavlakos/hamer, `fetch_demo_data.sh`,
file _DATA/hamer_ckpts/checkpoints/hamer.ckpt). The checkpoint embeds MANO-derived buffers, so
you must have accepted the MANO license (https://mano.is.tue.mpg.de) to use it.
"""
import argparse
import os
import sys
import time

from .assets import ensure_mano_license, torch_cache_bundle_dir


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="fasthamer-convert-torch",
                                 description="Convert the official hamer.ckpt into a fasthamer torch bundle.")
    ap.add_argument("--ckpt", required=True, help="path to hamer.ckpt")
    ap.add_argument("--out", default=None, help="bundle directory (default: the fasthamer cache)")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"],
                    help="storage precision of the network weights (default float16)")
    args = ap.parse_args(argv)
    try:
        import torch  # noqa: F401
    except ImportError:
        print("[fasthamer] the torch backend needs PyTorch: pip install torch  (see https://pytorch.org)", file=sys.stderr)
        return 1
    from .torch_model import convert_hamer_checkpoint

    if not os.path.isfile(args.ckpt):
        print(f"[fasthamer] no such checkpoint: {args.ckpt}", file=sys.stderr)
        return 1
    try:
        ensure_mano_license(interactive=sys.stdin.isatty())
    except RuntimeError as e:
        print(f"[fasthamer] {e}", file=sys.stderr)
        return 1
    out = args.out or torch_cache_bundle_dir()
    t0 = time.time()
    convert_hamer_checkpoint(args.ckpt, out, dtype=args.dtype)
    print(f"[fasthamer] wrote torch bundle to {out} in {time.time() - t0:.0f} s\n"
          f"Use it with: fasthamer.load(backend=\"torch\"{'' if out == torch_cache_bundle_dir() else f', model_dir={out!r}'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
