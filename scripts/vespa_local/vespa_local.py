import argparse

import os
import time

VESPA_VERSION = os.getenv('VESPA_VERSION', '8.396.18')  # default version baked into marqo-base:30


def start(args):
    os.system("docker rm -f vespa 2>/dev/null || true")

    os.system("docker run --detach "
              "--name vespa "
              "--hostname vespa-container "
              "--publish 8080:8080 --publish 19071:19071 --publish 2181:2181 --publish 127.0.0.1:5005:5005 "
              f"vespaengine/vespa:{VESPA_VERSION}")


def restart(args):
    os.system("docker restart vespa")


def deploy_config(args):
    os.system('vespa config set target local')
    here = os.path.dirname(os.path.abspath(__file__))

    max_retries = 10
    for attempt in range(1, max_retries + 1):
        print(f"Attempt {attempt}/{max_retries}: Deploying application package...")
        result = os.system(f'vespa deploy "{here}"')

        if result == 0:
            print("Deployment successful.")
            break
        else:
            print(f"Deployment failed. Retrying in 1 second...")
            time.sleep(1)
    else:
        print("Deployment failed after 10 attempts.")


def stop(args):
    os.system('docker stop vespa')


def main():
    parser = argparse.ArgumentParser(description="CLI for local Vespa deployment.")

    subparsers = parser.add_subparsers(title="modes", description="Available modes", help="Deployment modes",
                                       dest='mode')
    subparsers.required = True  # Ensure that a mode is always specified

    prepare_parser = subparsers.add_parser("start", help="Start local Vespa")
    prepare_parser.set_defaults(func=start)

    restart_parser = subparsers.add_parser("restart", help="Restart existing local Vespa")
    restart_parser.set_defaults(func=restart)

    deploy_parser = subparsers.add_parser("deploy-config", help="Deploy config")
    deploy_parser.set_defaults(func=deploy_config)

    stop_parser = subparsers.add_parser("stop", help="Stop local Vespa")
    stop_parser.set_defaults(func=stop)

    # Parse the command-line arguments and execute the corresponding function
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        # If no command was provided, print help information
        parser.print_help()


if __name__ == "__main__":
    main()
