# Developer guide
Thank you for contributing to Marqo! Contributions from the open source community help make Marqo be the tensor engine
you want. 

See [here](https://github.com/marqo-ai/marqo/blob/mainline/CONTRIBUTING.md#unit-tests) for how to run unit tests.

## Running Marqo locally (outside of docker) for development

There are two ways to run Marqo locally (outside of docker) for development: Option A. through `uvicorn`, 
Option B. through your IDE (e.g. PyCharm). 
We highly recommend using the Option A, as it allows you to set breakpoints and debug Marqo. 
Before running Marqo locally, you will need to do some preparations to set up Vespa locally.

### Preparations

1. Clone the Marqo Github repo and cd into it
```bash
git clone https://github.com/marqo-ai/marqo.git
cd marqo
```

2. Install Marqo dependencies
```bash
pip install -r requirements.txt
```

3. Pull and run the Vespa docker image
```bash
docker run --detach --name vespa --hostname vespa-tutorial \
  --publish 8080:8080 --publish 19071:19071 --publish 19092:19092 \
  vespaengine/vespa:latest
```

4. Install the vespa command line interface (CLI)
Check out the [Vespa CLI documentation](https://docs.vespa.ai/en/vespa-cli.html) for how to install the CLI.

5. Deploy a dummy application for Vespa docker image
```bash
vespa deploy scripts/vespa_dummy_app
```

Up to now. you should have a running Vespa docker image and a dummy application deployed to it. 
You can check this by visiting `http://localhost:8080` in your browser.

### Option A. Run the Marqo application locally (outside of docker) through IDE
Now you can run Marqo locally through your IDE (e.g. PyCharm) by following the steps below.

6. Open the Marqo project in your IDE (e.g. PyCharm) and go to the file `src/marqo/tensor_search/api.py`
7. Set up your debug configuration to run `api.py` with the following environment variables:
```
MARQO_ENABLE_BATCH_APIS=true;
MARQO_LOG_LEVEL=debug;
MARQO_MODELS_TO_PRELOAD=[];
VESPA_CONFIG_URL=http://localhost:19071;
VESPA_DOCUMENT_URL=http://localhost:8080;
VESPA_QUERY_URL=http://localhost:8080
```
8. Now you can Debug this file directly from your IDE (e.g. PyCharm) to start Marqo locally.
9. Set breakpoints in the project for better debugging experience.


### Option B. Run the Marqo application locally (outside of docker) through `uvicorn`
Finish the preparations above, then run the following command:

6. Set up the environment variables and run Marqo through `uvicorn`
```bash
export MARQO_ENABLE_BATCH_APIS=true
export MARQO_LOG_LEVEL=debug
export VESPA_CONFIG_URL=http://localhost:19071
export VESPA_DOCUMENT_URL=http://localhost:8080
export ESPA_QUERY_URL=http://localhost:8080
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
cd src/marqo/tensor_search
uvicorn api:app --host 0.0.0.0 --port 8882 --reload
```

### Notes:

#### Redis setup (Applicable for Options A and B)
Marqo uses redis to handle concurrency throttling. Redis is automatically set up when running Marqo in docker, but if you are running Marqo locally on your machine (Options A and B), you will have to set redis up yourself to enable throttling.

Note: This setup is optional. If you do not have redis set up properly, Marqo will still run as normal, but throttling will be disabled (you will see warnings containing `There is a problem with your redis connection...`). To suppress these warnings, disable throttling completely with:
```
export MARQO_ENABLE_THROTTLING='FALSE'
```

#### Installation
The redis-server version to install is redis 7.0.8. Install it using this command for Ubuntu 22.0.4:
```
apt-get update
apt-get install redis-server -y
```

If you are using an older version of Ubuntu, this may install an older version of redis. To get the latest redis version, run these commands instead:
```
apt install lsb-release
curl -fsSL https://packages.redis.io/gpg | gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/redis.list
apt-get update
apt-get install redis-server -y
``` 

#### Running redis
To start up redis, simply run the command:
```
redis-server /etc/redis/redis.conf
```

The `/etc/redis/redis.conf` configuration file should have been automatically created upon the redis installation step.


## Running Marqo in docker for development

### Option C. Build and run the Marqo as a Docker container
1. `cd` into the marqo root directory
2. Run the following command:
```bash
docker rm -f marqo &&
     DOCKER_BUILDKIT=1 docker build . -t marqo_docker_0 
     docker run --name marqo -p 8882:8882 marqo_docker_0
```

### Using Marqo with a GPU
Depending if you are running Marqo within Docker or not, there are different steps to take to use a GPU.

#### Using Marqo outside of Docker
Marqo outside Docker will rely on the system setup to use the GPU. If you can use a GPU normally with pytorch then it should be good to go. The usual caveats apply though, the CUDA version of pytorch will need to match that of the GPU drivers (see below on how to check).

#### Using Marqo within Docker
Currently, only CUDA based (Nvidia) GPU's are supported. If you have a GPU on the host machine and want to use it with Marqo, there are two things to do; 
1. Add the `--gpus all` flag to the `docker run` command. This command excluded from the above but will allow the GPU's to be used within Marqo. For example, in the steps B., C., and D., above `--gpus all` should be added after the 
`docker run --name marqo` part of the command, e.g. B. from above would become,
```bash
docker rm -f marqo &&
     DOCKER_BUILDKIT=1 docker build . -t marqo_docker_0 && 
     docker docker run --name --gpus all marqo -p 8882:8882 marqo_docker_0
```
note the `--gpus all` has been added.

2. Install [nvidia-docker2](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) which is required for the GPU to work with Docker. The [link](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) provided has instructions for installing it but it should consist of only a couple of steps (refer to the link for full details). The three steps below should install it for a Ubuntu based machine;  
```
$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

$ sudo apt-get update
$ sudo apt-get install -y nvidia-docker2
```
Once this is installed, you can include `--gpus all` in the `docker run` command to allow the GPU to be used within Marqo.

### Using Marqo on an AWS EC2 machine
#### (Note: This is not recommended for production use cases.)

1. Install docker

To install Docker (through terminal) go to the [Official Docker Website](https://docs.docker.com/engine/install/ubuntu/)

2. Set up SSH Config (to stop timeouts)

Edit the SSH config file with `nano ~/.ssh/config` then insert the line: `ServerAliveInterval 50`

3. Run Marqo

```bash
docker docker run --name  marqo -p 8882:8882 marqoai/marqo:latest
```

#### Troubleshooting
##### Drivers
In order for the GPU to be used within Marqo, the underlying host needs to have NVIDIA drivers installed. The current driver can be easily accessed by typing 

```
nvidia-smi
```

in a terminal. If there is not output then there may be something wrong with the GPU setup and installing or updating drivers may be necessary.  

##### CUDA 
Aside from having the correct drivers installed, a matching version of CUDA is required. The marqo Dockerfile comes setup to use CUDA 11.4.2 by default. The Dockerfile can be easily modified to support different versions of CUDA. Note, ONNX requires the system CUDA while pytorch relies on its own version of CUDA. 

##### Checking the status of your GPU and CUDA
To see if a GPU is available when using pytorch, the following can be used to check (from python);
```python
$ import torch
$ torch.cuda.is_available() # is a GPU available
$ torch.version.cuda        # get the CUDA version
$ torch.cuda.device_count() # get the number of devices
```
To check your driver and maximum CUDA version supported, type the following into the terminal;
```
nvidia-smi
```
Pytorch comes with its own bundled CUDA which allows many different CUDA versions to be used. Follow the [getting started](https://pytorch.org/get-started/locally/) to see how to install different versions of pytorch and CUDA.

## Extracting `openapi.json` (swagger API spec)
To get just the json, run this command (if Marqo is running locally)
```
curl http://localhost:8882/openapi.json
```
To get the human readable spec, visit `http://localhost:8882/docs`

## IDE tips

## PyCharm
Pydantic dataclasses are used in this project. By default, PyCharm can't parse initialisations of these dataclasses. 
[This plugin](https://plugins.jetbrains.com/plugin/12861-pydantic) can help.
