# Developer guide
Thank you for contributing to Marqo! Contributions from the open source community help make Marqo be the tensor engine
you want. 

See [here](https://github.com/marqo-ai/marqo/blob/mainline/CONTRIBUTING.md#unit-tests) for how to run unit tests.

## Select an option (from A-E) to get set up. In most cases, Option A is recommended. 

### Option A. Run the Marqo application locally (outside of docker):

1. Have a running Marqo-OS instance available to use. You can spin up a local instance with the following command
(if you are using an arm64 machine, replace `marqoai/marqo-os:0.0.3` with `marqoai/marqo-os:0.0.3-arm`):
```bash
docker run --name marqo-os -id -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" marqoai/marqo-os:0.0.3
```

2. Clone the github repo
```
git clone https://github.com/marqo-ai/marqo.git
```
3. Install marqo dependencies
```
cd marqo
pip install -r requirements.txt
```
4. Run the following command:
```bash
# if you are running Marqo-OS locally: 
export OPENSEARCH_URL="https://localhost:9200" 
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
cd src/marqo/tensor_search
uvicorn api:app --host 0.0.0.0 --port 8882 --reload
```
__Notes__:

- This is for marqo-os (Marqo OpenSearch) running locally. You can alternatively set
`OPENSEARCH_URL` to  a remote Marqo OpenSearch cluster 

### Option B. Build and run the Marqo as a Docker container, that creates and manages its own internal Marqo-OS 
1. `cd` into the marqo root directory
2. Run the following command:
```bash
docker rm -f marqo &&
     DOCKER_BUILDKIT=1 docker build . -t marqo_docker_0 
     docker run --name marqo --privileged -p 8882:8882 --add-host host.docker.internal:host-gateway marqo_docker_0
```

### Option C. Build and run the Marqo as a Docker container, connecting to Marqo-OS which is running on the host:
1. Run the following command to run Marqo-OS (if you are using an arm64 machine, replace `marqoai/marqo-os:0.0.3` with `marqoai/marqo-os:0.0.3-arm`):
```bash
docker run --name marqo-os -id -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" marqoai/marqo-os:0.0.3
```

2. `cd` into the marqo root directory
3. Run the following command:
```bash
docker rm -f marqo &&
     DOCKER_BUILDKIT=1 docker build . -t marqo_docker_0 && 
     docker run --name marqo --privileged -p 8882:8882 --add-host host.docker.internal:host-gateway \
         -e "OPENSEARCH_URL=https://localhost:9200" marqo_docker_0
```

__Notes__:

- This is for marqo-os (Marqo OpenSearch) running locally. You can alternatively set
`OPENSEARCH_URL` to  a remote Marqo OpenSearch cluster 
### Option D. Pull marqo from `hub.docker.com` and run it
```
docker rm -f marqo &&
    docker run --name marqo --privileged -p 8882:8882 --add-host host.docker.internal:host-gateway marqoai/marqo:latest
```

### Option E. Run marqo on arm64 (including M-series Macs) for development

1. Run marqo-os,
```
docker run --name marqo-os -id -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" marqoai/marqo-os:0.0.3-arm 
```

2. Clone the Marqo github repo (if not already done),
```
git clone https://github.com/marqo-ai/marqo.git
```

3. change into the Marqo directory,
```
cd marqo
```

4. Install some dependencies (requires [Homebrew](https://brew.sh/)),
```
brew install cmake;
brew install protobuf;
```

5. [Install rust](https://www.rust-lang.org/tools/install),
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs/ | sh;
```

6. Install marqo dependencies,
```
pip install -r requirements.txt
```

7. Change into the tensor search directory,
```
CWD=$(pwd)
cd src/marqo/tensor_search/
```
8. Run Marqo,
```
export OPENSEARCH_URL="https://localhost:9200" && 
    export PYTHONPATH="${PYTHONPATH}:${CWD}/src" &&
    uvicorn api:app --host 0.0.0.0 --port 8882 --reload
```

### Using Marqo with a GPU
Depending if you are running Marqo within Docker (steps B., C. and D.) or not (step A.) will determine if you need to do anything to use a GPU with Marqo.

#### Using Marqo outside of Docker
Marqo outside Docker will rely on the system setup to use the GPU. If you can use a GPU normally with pytorch then it should be good to go. The usual caveats apply though, the CUDA version of pytorch will need to match that of the GPU drivers (see below on how to check).

#### Using Marqo within Docker
Currently, only CUDA based (Nvidia) GPU's are supported. If you have a GPU on the host machine and want to use it with Marqo, there are two things to do; 
1. Add the `--gpus all` flag to the `docker run` command. This command excluded from the above but will allow the GPU's to be used within Marqo. For example, in the steps B., C., and D., above `--gpus all` should be added after the 
`docker run --name marqo` part of the command, e.g. B. from above would become,
```bash
docker rm -f marqo &&
     DOCKER_BUILDKIT=1 docker build . -t marqo_docker_0 && 
     docker run --name marqo --gpus all --privileged -p 8882:8882 --add-host host.docker.internal:host-gateway \
         -e "OPENSEARCH_URL=https://localhost:9200" marqo_docker_0
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
Once this is installed, one of the previous Docker commands can be run (either step B., C., or D.).

### Using Marqo on an AWS EC2 machine
#### (Note: This is not recommended for production use cases.)

1. Install docker

To install Docker (through terminal) go to the [Official Docker Website](https://docs.docker.com/engine/install/ubuntu/)

2. Set up SSH Config (to stop timeouts)

Edit the SSH config file with `nano ~/.ssh/config` then insert the line: `ServerAliveInterval 50`

3. Run marqo-os
```
sudo docker rm -f marqo-os; sudo docker run -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" marqoai/marqo-os:0.0.3-arm
```

4. Run marqo with a set `OPENSEARCH_URL`
```
sudo docker rm -f marqo; sudo docker run --name marqo -it --privileged -p 8882:8882 --add-host host.docker.internal:host-gateway -e "OPENSEARCH_URL=https://localhost:9200" marqoai/marqo:latest
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

