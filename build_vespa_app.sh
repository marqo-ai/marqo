#!/bin/bash

# Install maven
wget https://dlcdn.apache.org/maven/maven-3/3.9.8/binaries/apache-maven-3.9.8-bin.tar.gz -P /temp
tar xf /temp/apache-maven-3.9.8-bin.tar.gz -C /opt
ln -s /opt/apache-maven-3.9.8 /opt/maven

touch /etc/profile.d/maven.sh
bash -c 'echo "export JAVA_HOME=/usr/lib/jvm/jre-openjdk" > /etc/profile.d/maven.sh'
bash -c 'echo "export M2_HOME=/opt/maven" >> /etc/profile.d/maven.sh'
bash -c 'echo "export MAVEN_HOME=/opt/maven" >> /etc/profile.d/maven.sh'
bash -c 'echo "export PATH=\${M2_HOME}/bin:\${PATH}" >> /etc/profile.d/maven.sh'

chmod +x /etc/profile.d/maven.sh
source /etc/profile.d/maven.sh

# Build Hybrid Searcher application and Jar
echo "Building hybrid searcher jar"
cd /app/scripts/vespa_local
mvn clean package