# Vespa on EKS script
Usage:
1. Install dependencies: `pip install -r requirements.txt`
2. Create a config file `my_config.json` by copying and editing `sample_config.json`
2. Run `python vespa_cluster.py prepare my_config.json`
3. Run `python vespa_cluster.py deploy-eks` to create the EKS cluster
4. Run `python vespa_cluster.py deploy-services my_config.json` to deploy Vespa services
5. Run `python vespa_cluster.py deploy-config my_config.json` to deploy the Vespa application package 
6. Run `python vespa_cluster.py get-endpoints my_config.json` to get the endpoints of the Vespa cluster

Notes: 
* You may need to wait a few minutes for the services to be ready before deploying the Vespa application package
* Vespa application package deployment will complain that `service 'query' is unavailable: services have not converged`.
This is expected as pod endpoints are not directly reachable from your machine.

