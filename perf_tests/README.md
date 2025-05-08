# Marqo performance Tests
We use [locust](https://docs.locust.io/en/stable/what-is-locust.html) to run performance test for Marqo

## Setup to run locally
### Preparation
```shell
# Create a new virtual env (so the dependency does not clash with src folder
python -m venv ./venv_perf_tests
source ./venv-perf-tests/bin/activate

cd perf_tests
pip install -r requirements.txt
```

### Start Marqo
Start a Marqo server in a container or in your local IDE listening on 8882 port.
```shell
docker run --name marqo -d -p 8882:8882 -e MARQO_MODELS_TO_PRELOAD='["hf/e5-base-v2"]' marqoai/marqo
```

### Run performance test locally
```shell
# this will use the default config in locust.conf file
locust

# Alternatively you can specify CLI params to override the default settings
locust -u <user_count> -r <spawn-rate> -t <duraion> -H <host> -f <test_file> 

# When run locally, by default it creates an index `locust-test` with `hf/e5-base-v2` model,
# You can specify the name of the index or model by using env vars
MARQO_INDEX_NAME=<index_name> MARQO_INDEX_MODEL_NAME=<model_name> locust 

# You can also run against a Marqo Cloud instance by providing the host and API key
MARQO_INDEX_NAME=<index_name> MARQO_CLOUD_API_KEY=<your api key> locust -H <host>

# After the run, a test report will be generated to report/report.html file
```

## Trigger run in Github
TODO add after the GH action is merged to master

## Develop new test cases

Locust test cases are all written in plain python code. You can add new test scenarios in a separate 
locust python file. The current project structure is 

```text
| - common/      # reusable utilities for all test cases
| - locustfiles/ # reusable test cases (TaskSet)
| - test_suite_1.py # test scenario 1
\ - test_suite_2.py # test scenario 2
```

Please use `random_index_and_tensor_search.py` as an example.
More examples can be find in [Locust documents](https://docs.locust.io/en/stable/writing-a-locustfile.html)

### Local IDE setup
If you are using Pycharm, you will need to
* Choose the right Python interpreter in Settings -> Project -> Python Interpreter (use the one in the new virtual env)
* Uncheck `src` folder as Sources in Settings -> Project -> Project Structure (this allows you to see py-marqo)
* Check `Gevent compatible` option in Settings -> Python Debugger (this allows you to debug your test cases locally)

```python
# This is an example to debug SearchUser test case locally
from locust import run_single_user

if __name__ == "__main__":
    run_single_user(SearchUser)
```

### Add new dependencies
We use pip-tools to generate `requirements.txt` file from `requirements.in`. Please do not edit
`requirements.txt` file directly. Instead, add the direct dependency to `requirements.in`, and run

```shell
pip install pip-tools
pip-compile requirements.in
```

