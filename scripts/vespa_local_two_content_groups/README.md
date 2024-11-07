# Run Vespa cluster (Multi-group) locally

## Steps 
1. Set the Vespa version in `.env` file, for example `VESPA_VERSION=8.396.18`, if not set, `latest` will be used
2. `docker compose up -d`
3. Check if the config server is up and running

```shell
# config server
curl http://localhost:19071/state/v1/health
```

4. Deploy the vespa application

```shell
zip -r - . -x README.md .env .gitignore "*.yml" | \
  curl --header Content-Type:application/zip --data-binary @- \
  http://localhost:19071/application/v2/tenant/default/prepareandactivate
```

5. Check Convergence of the Vespa app

```shell
curl http://localhost:19071/application/v2/tenant/default/application/default/environment/prod/region/default/instance/default/serviceconverge | jq .
```

6. Check if other servers are up and running

```shell
# content servers
curl http://localhost:19107/state/v1/health
curl http://localhost:19108/state/v1/health

# container server
curl http://localhost:8080/state/v1/health
curl http://localhost:8080/ApplicationStatus
```


## References
- [multinode-HA cluster](https://github.com/vespa-engine/sample-apps/tree/master/examples/operations/multinode-HA)
- [Vespa Sizing example](https://docs.vespa.ai/en/performance/sizing-examples.html)