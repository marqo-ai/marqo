import os
import subprocess
import time
import typing
import pathlib
import tempfile
import yaml

root_project_dir = pathlib.Path(__file__).resolve().parent.parent.parent.parent.parent.parent.parent
compose_file = os.path.join(root_project_dir, "compose.yaml")


def disallow_environments(disallowed_configurations: typing.List[str]):
    """This construct wraps a test to ensure that it does not run for disallowed
    testing environments.

    It figures by examining the "TESTING_CONFIGURATION" environment variable.

    Args:
        disallowed_configurations: if the environment variable
        "TESTING_CONFIGURATION" matches a configuration in
        disallowed_configurations, then the test will be skipped
    """

    def decorator(function):
        def wrapper(*args, **kwargs):
            if os.environ["TESTING_CONFIGURATION"] in disallowed_configurations:
                return
            else:
                result = function(*args, **kwargs)
                return result

        return wrapper

    return decorator


def allow_environments(allowed_configurations: typing.List[str]):
    def decorator(function):
        def wrapper(*args, **kwargs):
            if os.environ["TESTING_CONFIGURATION"] not in allowed_configurations:
                return
            else:
                result = function(*args, **kwargs)
                return result

        return wrapper

    return decorator


def classwide_decorate(decorator, allowed_configurations):
    def decorate(cls):
        for method in dir(cls):
            if method.startswith("test"):
                setattr(cls, method, (decorator(allowed_configurations))(getattr(cls, method)))
        return cls

    return decorate


def rerun_marqo_with_env_vars(env_vars: dict[str, str], calling_class: str = "", target_service: str = "api"):
    """
        Given a list of env vars / flags, stop and rerun Marqo using the start script appropriate
        for the current test config

        Ensure that:
        1. Flags are separate items from variable itself (eg, ['-e', 'MARQO_MODELS_TO_PRELOAD=["hf/all_datasets_v4_MiniLM-L6"]'])
        2. Strings (individual items in env_vars list) do not contain ' (use " instead)
        -> single quotes cause some parsing issues and will affect the test outcome
    """

    if calling_class not in ["TestEnvVarChanges", "TestBackendRetries"]:
        raise RuntimeError(
            f"Rerun Marqo function should only be called by `TestEnvVarChanges` "
            f"to ensure other API tests are not affected. Given calling class is {calling_class}"
        )

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".env") as fp:

        with open(compose_file, 'r') as compose_fp:
            compose_content = yaml.safe_load(compose_fp)

        updated_compose_data = compose_content.copy()

        targe_service_file = updated_compose_data['services'][target_service]
        for key, value in env_vars.items():
            if not 'environment' in targe_service_file:
                targe_service_file['environment'] = {}
            targe_service_file['environment'][key] = value

        yaml.dump(updated_compose_data, fp)
        fp.flush()
        temp_path = pathlib.Path(fp.name).absolute()

        run_process = subprocess.Popen(
            [
                "docker",  # command: run
                "compose",
                "-f",
                temp_path,
                "up",
                "-d",
                "--force-recreate",
                target_service
            ],
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        universal_newlines = True
        )
        # Wait for the process to complete
        run_process.wait()
        return True


def rerun_marqo_with_default_config(calling_class: str = ""):
    # Do not send any env vars
    # This should act like running the start script at the beginning
    rerun_marqo_with_env_vars(env_vars={}, calling_class=calling_class)


docker_log_failure_message = "Failed to fetch docker logs for Marqo"


def attach_docker_logs(container_name: str, log_collection: typing.List, start_time: str = None) -> None:
    """Fetches the Docker logs of a specified container and stores them in a provided list.

    Args:
        container_name (str): Name of the Docker container whose logs are to be
            fetched.
        log_collection (List): A list which its first element is used to store
            the fetched logs or error messages. A list is used to store the
            logs, rather than returning them, so this function can be used in a
            thread.
        start_time (str): A string representing the start time stamp to get docker logs
            must be in the format: "%Y-%m-%dT%H:%M:%S"
    """

    commands = ["docker", "logs", container_name]

    if start_time != None:
        commands.append(f"--since={start_time}")

    completed_process = subprocess.run(
        commands,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    if completed_process.returncode == 0:
        log_collection.append(completed_process.stdout)
    else:
        log_collection.append(
            f"{docker_log_failure_message}. "
            f"Failed with error: {completed_process.stderr}")


def retrieve_docker_logs(
        container_name: str,
        start_time: str = None,
) -> str:
    """Returns docker logs as a string, for a specific container

    Args:
        container_name (str): Name of the Docker container whose logs are to be checked. Defaults to 'marqo'.
    Returns:
        A str which is the docker logs for the container.
    Raises:
        RuntimeError: If fetching logs fails or times out.
    """
    # a 1-elem mutable object to save the docker logs to:
    log_collection = []

    docker_log_fetcher = attach_docker_logs

    docker_log_fetcher(container_name=container_name, log_collection=log_collection, start_time=start_time)

    if not log_collection:
        raise RuntimeError(
            "Fetching logs timed out or failed. log_collection is empty.")

    if docker_log_failure_message in log_collection[0]:
        raise RuntimeError(f"{docker_log_fetcher.__name__} encountered an "
                           f"error retrieving docker logs. {log_collection[0]}")

    return log_collection[0]


def control_marqo_os(
        container_name: str = "marqo-os",
        command: str = "start",
):
    """Stops a Marqo OS container. If Setup is DIND, This executes a command on the marqo container.

    Args:
        container_name (str): Name of the Docker container to stop. Defaults to 'marqo-os'.
    """
    docker_command = f"docker {command} {container_name}"

    time.sleep(10)
    if "DIND" in os.environ["TESTING_CONFIGURATION"]:
        command_output = subprocess.run(
            f"docker exec marqo sh -c '{docker_command}'",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
    else:
        command_output = subprocess.run(
            docker_command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
    time.sleep(10)
    print(command_output.stdout)
