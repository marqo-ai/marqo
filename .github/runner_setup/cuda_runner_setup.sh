# When setting up a new runner AMI, paste these commands in to install the necessary software
# Then create an AMI from your instance

# Run the AMD setup commands first
# Then these:

sudo apt-get update
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)       && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg       && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list |             sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' |             sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get install -y nvidia-docker2

# Check if drivers are properly installed with nvidia-smi
nvidia-smi

# If GPU not find, try installing drivers
sudo apt install nvidia-utils-520
sudo apt install nvidia-driver-515 nvidia-dkms-515
sudo reboot