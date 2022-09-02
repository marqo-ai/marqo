
### A. Run the Marqo application locally (outside of docker):

1. `cd` into `src/marqo/tensor_search`
2. Run the following command:
```bash
# if you are running OpenSearch locally. 
export OPENSEARCH_URL="https://localhost:9200" && 
    export PYTHONPATH="${PYTHONPATH}:<your path here>/marqo/src" &&
    uvicorn api:app --host 0.0.0.0 --port 8882 --reload
```
__Notes__:

- This is for marqo-os (Marqo OpenSearch) running locally. You can alternatively set
`OPENSEARCH_URL` to  a remote Marqo OpenSearch cluster 
- 


### B. Build and run the Marqo as a Docker container, that creates and manages its own internal OpenSearch 
1. `cd` into the marqo root directory
2. Run the following command:
```bash
docker rm -f marqo &&
     DOCKER_BUILDKIT=1 docker build . -t marqo_docker_0 && 
     docker run --name marqo --privileged -p 8882:8882 --add-host host.docker.internal:host-gateway marqo_docker_0
```

### C. Build and run the Marqo as a Docker container, connecting to OpenSearch which is running on the host:
1. `cd` into the marqo root directory
2. Run the following command:
```bash
docker rm -f marqo &&
     DOCKER_BUILDKIT=1 docker build . -t marqo_docker_0 && 
     docker run --name marqo --privileged -p 8882:8882 --add-host host.docker.internal:host-gateway \
         -e "OPENSEARCH_URL=https://localhost:9200" marqo_docker_0
```


### D. Pull marqo from `hub.docker.com` and run it
```
docker run --name marqo --privileged -p 8882:8882 --add-host host.docker.internal:host-gateway marqoai/marqo:0.0.1
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
