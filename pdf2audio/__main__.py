"""Enables `python -m pdf2audio`; delegates to the CLI (installed as `pdf2audio`)."""

from pdf2audio.cli import main

if __name__ == "__main__":
    main()
