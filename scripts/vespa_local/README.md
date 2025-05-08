# Setting up Vespa locally
When running Marqo or the unit test suite locally, a Vespa node or cluster needs to be running. To assist with this, 
this directory comes with scripts to set up either a single node (1 container) or multinode-HA Vespa on your machine.

### Set Vespa version
- By default, this script will use Vespa 8.431.32, as defined in `vespa_local.py`. To change it, set the `VESPA_VERSION`
variable to the desired version. For example:
```commandline
export VESPA_VERSION="latest"
```
### Set Vespa max disk utilization
**NOTE:** Not recommended for production use. This is only for local development.
- By default, Vespa has limit of 0.75 (75%) disk utilization. To change it, set the `VESPA_DISK_USAGE_LIMIT` variable to float
value between 0 and 1. For example:
```commandline
export VESPA_DISK_USAGE_LIMIT=0.9
```
## Single Node Vespa (default & recommended)
- Runs 1 Vespa container on your machine. This serves as the config, api, and content node.
- This is equivalent to running Vespa with 0 replicas and 1 shard.
- Start with this command:
```commandline
python vespa_local.py start
```
- This will run the Vespa docker container then copy the `services.xml` file from the `singlenode/` directory to 
this directory. This will be bundled into the Vespa application upon deployment.

## Multi-node Vespa
- Runs a Vespa cluster with the following nodes:
  - 3 config nodes
  - `m` content nodes, where `m` is `number_of_shards * (1 + number_of_replicas)`
  - `n` API nodes, where `n` is `max(2, number_of_content_nodes)`
- For example, with 2 shards and 1 replica, it will run 4 content nodes and 2 API nodes.
- Start with this command:
```commandline
python vespa_local.py start --Shards 2 --Replicas 1
```

## Deployment
- After starting the Vespa node(s), you can deploy the Vespa application with the files in this directory using:
```commandline
python vespa_local.py deploy-config
```
- For single node, you can check for readiness using:
```
curl -s http://localhost:19071/state/v1/health
```
- For multi-node, the start script will output a list of URLs corresponding to the API and content nodes.
You can curl each one to check for readiness.

## Other Commands
### Stop Vespa
```commandline
python vespa_local.py stop
```
### Restart Vespa
```commandline
python vespa_local.py restart
```

## Notes
- When running other commands in this script (stop, restart), it will check for the presence of a container named 
`vespa`, and will assume setup is single node if it finds one. If not, it will assume setup is multi-node.
- For multi-node, expect config and API nodes to take ~1gb of memory, while content nodes take ~500mb each. Adjust your
resource allotment accordingly.