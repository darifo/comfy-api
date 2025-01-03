import argparse
import enum
import os

parser = argparse.ArgumentParser()

parser.add_argument("--listen", type=str, default="0.0.0.0", metavar="IP", nargs="?", const="0.0.0.0,::", help="Listen on IP.")
parser.add_argument("--port", type=int, default=9696, help="Set the listen port.")

parser.add_argument("--workers", type=int, default=1, help="Set the number of workers.")

parser.add_argument("--verbose", default='INFO', const='DEBUG', nargs="?", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Set the logging level')
parser.add_argument("--log-stdout", action="store_true", help="Send normal process output to stdout instead of stderr (default).")


args = parser.parse_args()