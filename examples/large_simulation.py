"""Reserved entry point for large-array simulations; backend not implemented."""

import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    parser.exit(
        2, "Large-array simulation is not implemented. See docs/large-simulations.md.\n"
    )


if __name__ == "__main__":
    main()
