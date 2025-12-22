# Developer guide
Thank you for contributing to Marqo! Contributions from the open source community help make Marqo be the tensor engine
you want. 

See [here](https://github.com/marqo-ai/marqo/blob/mainline/CONTRIBUTING.md#unit-tests) for how to run unit tests.

## Running Marqo locally (outside of docker) for development

There are two ways to run Marqo locally (outside of docker) for development: Option A. through `uvicorn`, 
Option B. through your IDE (e.g., PyCharm). 
We highly recommend using the Option A, as it allows you to set breakpoints and debug Marqo. 
Before running Marqo locally, you will need to do some preparations to set up Vespa locally.

### Preparations

1. Clone the Marqo GitHub repo and marqo-base repo
```bash
git clone https://github.com/marqo-ai/marqo.git
git clone https://github.com/marqo-ai/marqo-base.git
```

2. Install Marqo dependencies. Please choose the appropriate requirements file based on your environment.
**For AMD machine**, no matter you have a GPU or not: 
```bash
pip install -r marqo-base/requirements/amd64-gpu-requirements.txt  # For AMD machine, no matter you have a GPU or not
```
**For ARM machine**:
```bash
pip install -r marqo-base/requirements/arm64-requirements.txt  # For ARM machine
```
The above requirements are for running Marqo locally. If you need to run tests, you need:
```bash
pip install -r marqo/requirements.dev.txt  # For test purposes
```

3. Pull and run the Vespa docker image
```bash
docker run --detach --name vespa --hostname vespa-tutorial \
  --publish 8080:8080 --publish 19071:19071 \
  vespaengine/vespa:latest
```

4. Configure Vespa by deploying to the provided application package `scripts/vespa_local`
```bash
(cd scripts/vespa_local && zip -r - * | curl --header "Content-Type:application/zip" --data-binary @- http://localhost:19071/application/v2/tenant/default/prepareandactivate)
```
You can verify that Vespa has been set up correctly by visiting `http://localhost:8080` in your browser.

5. We need to install Java Development Kit and Maven to build the jar file for the custom searchers.

Install Java Development Kit: 

You can install the Java Development Kit (JDK) by following the instructions [here](https://docs.oracle.com/en/java/javase/22/install/overview-jdk-installation.html). 
Post installation, remember to set JAVA_HOME Environment Variable and to add Java to the PATH Environment Variable.
You can verify the installation by running the following command:
```bash
java -version
```

Install Maven: 

You can install Maven by following the instructions [here](https://maven.apache.org/install.html).
Similar to setting Java, you need to add Maven to the PATH Environment Variable.
Verify Maven installation by running the following command: 
```bash
mvn -version
```

6. After this you need to create a JAR file. To do that, go into the Vespa directory in your local Marqo repository by doing `cd vespa`, and run 
```bash
mvn clean package
```
After running this command, you will see that a target folder gets created in the vespa directory, which contains a JAR file called marqo-custom-searchers-deploy.jar. This JAR file is used to deploy the custom searchers to Vespa.

### Option A. Run the Marqo application locally (outside of docker) through IDE
Now you can run Marqo locally through your IDE by following the steps below.

7. Open the Marqo project in your IDE and go to the file `src/marqo/tensor_search/api.py`
8. Set up your [debug configuration](https://www.jetbrains.com/help/pycharm/creating-run-debug-configuration-for-tests.html)
to run `api.py` with the following environment variables:
```
MARQO_ENABLE_BATCH_APIS=true
MARQO_LOG_LEVEL=debug
MARQO_MODELS_TO_PRELOAD=[]
VESPA_CONFIG_URL=http://localhost:19071
VESPA_DOCUMENT_URL=http://localhost:8080
VESPA_QUERY_URL=http://localhost:8080
```
9. Now you can debug this file directly from your IDE (e.g., PyCharm) to start Marqo locally.
10. Set breakpoints in the project for better debugging experience.


### Option B. Run the Marqo application locally (outside of docker) through `uvicorn`
Finish the preparations above, then run the following command:

7. Set up the environment variables and run Marqo through `uvicorn`
```bash
export MARQO_ENABLE_BATCH_APIS=true
export MARQO_LOG_LEVEL=debug
export VESPA_CONFIG_URL=http://localhost:19071
export VESPA_DOCUMENT_URL=http://localhost:8080
export VESPA_QUERY_URL=http://localhost:8080
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
`docker run --name marqo` part of the command, e.g., B. from above would become,
```bash
docker rm -f marqo &&
     DOCKER_BUILDKIT=1 docker build . -t marqo_docker_0 && 
     docker run --name marqo --gpus all -p 8882:8882 marqo_docker_0
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
docker run --name  marqo -p 8882:8882 marqoai/marqo:latest
```

#### Troubleshooting
##### Drivers
In order for the GPU to be used within Marqo, the underlying host needs to have NVIDIA drivers installed. The current driver can be easily accessed by typing 

```
nvidia-smi
```

in a terminal. If there is not output then there may be something wrong with the GPU setup and installing or updating drivers may be necessary.  

##### CUDA 
Aside from having the correct drivers installed, a matching version of CUDA is required. The marqo Dockerfile comes setup to use CUDA 11.4.2 by default. The Dockerfile can be easily modified to support different versions of CUDA. 

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
