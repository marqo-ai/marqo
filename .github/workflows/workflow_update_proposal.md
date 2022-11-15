# Pipeline Update Proposal
## Overview
Marqo is having issues with unit & API testing (tests failing when they should pass, passing when they should fail, etc).

This could be due to: (1) runners not being powerful enough, (2) runners not being ephemeral, (3) all tests being executed in the same workflow.

Additionally, we need working test environments for ARM64 and CUDA machines.

## Proposed Design
### Workflow-generated EC2 Runners
Instead of github-hosted runners, custom AWS EC2 images will be made and used to instantiate ephemeral runner instances with the `ec2-github-runner` [library](https://github.com/machulav/ec2-github-runner).

1. EC2 instances will be set up (AMD64, ARM64, and CUDA) with `docker` and `git` installed and `docker` service enabled. required tools for testing will also be installed. (TODO: what exactly should be installed?).

AMD instance: t3.xlarge, 4CPUS, 16gbRAM, 50gb storage

ARM instance: a1.xlarge, 4CPUs, 8gbRAM, 50gb storage  

### Setting up AMD64 & ARM64 Image
- Install Docker with https://docs.docker.com/engine/install/ubuntu/

- Enable Docker with:
```bash
sudo systemctl enable docker.service  
sudo systemctl enable containerd.service
```

- Install git with `sudo apt-get install git`

### Setting up CUDA Image
- Follow all the steps for AMD64 and ARM64

- Install `nvidia-docker` like this:
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
```

- Install latest Nvidia Driver with:
```
sudo apt install -y nvidia-driver-515 nvidia-dkms-515
```

- Reboot the system with:
```
sudo reboot
```



2. EC2 images (AMI) will then be created from these instances.

3. Appropriate permissions will be set up on AWS/github.

4. Instances can now be spun up and down within the workflow itself using generated image ID.

Runners are started like this:
```yml
name: Start EC2 runner
        id: start-ec2-runner
        uses: machulav/ec2-github-runner@v2
        with:
          mode: start
          github-token: ${{ secrets.GH_PERSONAL_ACCESS_TOKEN }}
          ec2-image-id: ami-123
          ec2-instance-type: t3.nano
          subnet-id: subnet-123
          security-group-id: sg-123
          iam-role-name: my-role-name # optional, requires additional permissions
          aws-resource-tags: > # optional, requires additional permissions
            [
              {"Key": "Name", "Value": "ec2-github-runner"},
              {"Key": "GitHubRepository", "Value": "${{ github.repository }}"}
            ]
```

Runners are stopped like this:
```yml
name: Stop EC2 runner
        uses: machulav/ec2-github-runner@v2
        with:
          mode: stop
          github-token: ${{ secrets.GH_PERSONAL_ACCESS_TOKEN }}
          label: ${{ needs.start-runner.outputs.label }}
          ec2-instance-id: ${{ needs.start-runner.outputs.ec2-instance-id }}
```


Possible problems:
1. Speed could be an issue, since we create and stop new runners every workflow.

### Alternative: Self-hosted Existing EC2 Runners
Someone with repo/organization/enterprise-level permissions would add existing EC2 instances under "New Runners" in the github orgbanization settings. See steps [here](https://docs.github.com/en/actions/hosting-your-own-runners/adding-self-hosted-runners).




### Separate workflows per environment
The current setup has 1 workflow file: `CI.yml`, which runs 4 test environments: 
```yml
py3-local_os_unit_tests_w_requirements
py3-local_os
py3-dind_os
py3-s2search
```
The proposed solution would separate these environments into different workflows. Each workflow would run tests on a different environment.

These workflows will be run after every merge to main. The `tox.ini` file will be modified so workflows can be run against any branch of choice.

1. Unit Tests: `unit_test_CI.yml`
```yml
 - name: Run Unit Tests
    run: tox -e py3-local_os_unit_tests_w_requirements
```

2. Integration Tests on LOCAL OS: `local_os_CI.yml`
```yml
 - name: Run Integration Tests - local_os
    run: tox -e py3-local_os
```

3. Integration Tests on DIND OS: `dind_os_CI.yml`
```yml
 - name: Run Integration Tests - dind_os
    run: tox -e py3-dind_os
```

4. Integration Tests on S2SEARCH BACKEND: `s2search.yml`
```yml
 - name: Run Integration Tests - s2search
    run: tox -e py3-s2search
```

5. Integration Tests on ARM LOCAL OS: `arm64_local_os.yml`
```yml
# Start up ARM runner first
 - name: Run Integration Tests - ARM64 local_os
    run: tox -e py3-arm64_local_os
```

6. Integration Tests on CUDA LOCAL OS: `cuda_local_os.yml`
```yml
# Start up CUDA runner first
 - name: Run Integration Tests - CUDA local_os
    run: tox -e py3-cuda_local_os
```
A test environment for CUDA and script for running marqo on CUDA will have to be set up. 

TODO: What other environments should be run? Should unit tests be done on ARM and CUDA?

Possible problems:
1. Speed again, since every environment test would set up a new runner and install all requirements.