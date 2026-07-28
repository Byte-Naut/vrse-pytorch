"""Phase 3B / T0: freeze the C-MAPSS FD002 input manifest.

This script does not train a model.  It accepts either the original extracted
directory or an archive and records file-level hashes without asserting that a
locally repacked archive is an official NASA artifact.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from experiments.phase3_common import MANIFEST, PREPARED, ROOT
from src.phase3_cmapss import NASA_RESOURCE_PAGE, download_zip, save_prepared


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path, default=ROOT / "data" / "CMAPSSData",
        help="Original extracted C-MAPSS directory or a C-MAPSS ZIP archive.",
    )
    parser.add_argument("--zip", type=Path, help="Deprecated alias for --source.")
    parser.add_argument(
        "--source-origin", default="unknown",
        choices=(
            "unknown",
            "user-attested-official-download",
            "user-attested-original-extraction",
            "local-repack",
            "downloaded-user-supplied-url",
        ),
        help="Provenance claim supplied by the operator; hashes remain the authority.",
    )
    parser.add_argument(
        "--download-url",
        help="Optional direct official ZIP URL. The NASA landing/resource HTML URL is not a ZIP.",
    )
    args = parser.parse_args()
    source = args.zip if args.zip is not None else args.source
    if not source.exists():
        if not args.download_url:
            raise FileNotFoundError(
                f"Missing {source}. Download C-MAPSS from {NASA_RESOURCE_PAGE} "
                "or pass --download-url with a direct official ZIP URL."
            )
        source = ROOT / "data" / "CMAPSSData.zip"
        download_zip(args.download_url, source)
        args.source_origin = "downloaded-user-supplied-url"
    manifest = save_prepared(source, PREPARED, MANIFEST, args.source_origin)
    print(f"Prepared: {PREPARED}")
    print(f"Manifest: {MANIFEST}")
    print(
        "Regimes: ID={id_regime}, NEW={new_regime}, UNKNOWN={unknown_regime}".format(
            **manifest
        )
    )


if __name__ == "__main__":
    main()
