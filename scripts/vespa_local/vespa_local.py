import argparse
import json

import os
import shutil
import re
import subprocess


def start(args):
    os.system("docker rm -f vespa 2>/dev/null || true")

    os.system("docker run --detach "
              "--name vespa "
              "--hostname vespa-container "
              "--publish 8080:8080 --publish 19071:19071 "
              "--memory=\"32g\" "
              "vespaengine/vespa")


def restart(args):
    os.system("docker restart vespa")


def deploy_config(args):
    os.system('vespa config set target local')
    here = os.path.dirname(os.path.abspath(__file__))
    os.system(f'vespa deploy {here}')


def stop(args):
    os.system('docker stop vespa')


def main():
    parser = argparse.ArgumentParser(description="CLI for local Vespa deployment.")

    subparsers = parser.add_subparsers(title="modes", description="Available modes", help="Deployment modes",
                                       dest='mode')
    subparsers.required = True  # Ensure that a mode is always specified

    prepare_parser = subparsers.add_parser("start", help="Start local Vespa")
    prepare_parser.set_defaults(func=start)

    prepare_parser = subparsers.add_parser("restart", help="Restart existing local Vespa")
    prepare_parser.set_defaults(func=restart)

    eks_parser = subparsers.add_parser("deploy-config", help="Deploy config")
    eks_parser.set_defaults(func=deploy_config)

    clean_parser = subparsers.add_parser("stop", help="Stop local Vespa")
    clean_parser.set_defaults(func=stop)

    # Parse the command-line arguments and execute the corresponding function
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        # If no command was provided, print help information
        parser.print_help()


if __name__ == "__main__":
    main()
