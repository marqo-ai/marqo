# When setting up a new runner AMI, paste these commands in to install the necessary software
# Then create an AMI from your instance

# Install and enable docker
sudo apt-get remove docker docker-engine docker.io containerd runc
sudo apt-get install     ca-certificates     curl     gnupg     lsb-release
sudo apt-get update
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin

sudo systemctl enable docker.service
sudo systemctl enable containerd.service

# Install git
sudo apt-get install git

