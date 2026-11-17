"""Package API source without changing the existing Python environment."""

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Destination .pyz archive")
    output = parser.parse_args().output
    output.parent.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parent
    with ZipFile(str(output), "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(source.glob("*.py")):
            if path.name != "build.py":
                archive.write(str(path), "backend/" + path.name)
        archive.writestr("__main__.py", "import runpy\nrunpy.run_module('uvicorn', run_name='__main__')\n")
    print("Built {}".format(output))


if __name__ == "__main__":
    main()
