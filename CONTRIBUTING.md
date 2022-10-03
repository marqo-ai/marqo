Thank you for your interest in making tensor search more accessible!

---

## Ways to contribute

### Bug reports
You can submit bug reports on our GitHub repo

### Contributing code
We welcome contributions to the codebase. Here are some coding guidelines to follow:
* Where possible, we explicitly state the arg names when calling a function. 
  This makes refactoring easier. If possible, use `func(a=1, b=2)` rather than `func(1, 2)`

### Error usage
- Errors raised that concern non-user-facing functionality
(for example, related to vectors), should raise an `InternalError` or its subclass
- Errors arising from calls S2 Inference's API should raise an `S2InferenceError`    


### Semantic Versioning
We use semantic versioning. 
- We do regular minor releases. These are releases that are additive and don't break existing functionality. These add extent the API. During a minor release we bump the minor version number: 0.41.24 -> 0.42.24
- In-between minor releases, we release bug-fixes by bumping the patch number: 0.34.6 -> 0.34.7
- If there are any breaking changes, there will be a major release: 5.39.55 -> 6.39.55
- Note that pre-version 1.x.x, because this is the pre-release stage, breaking functionality may be introduced in minor versions
- Also, pre-version 1.x.x, patches to the docker image can be pushed without updating the docker tag

### Releasing changes
- Run unit tests and ensure they all pass (read the testing section below for more details)
- Generate a pull request to the `mainline` branch. These will be reviewed before merging
- After merging to `mainline`, please delete the branch with the pull request
- For minor releases, please record changes in  `RELEASE.md`
- For minor releases that change the API, update py-marqo to make those changes accessible. 
- To release a change to py-marqo, bump py-marqo's version (major.minor.patch, see below for more details), run all the tests, build the package and push to it PyPi. py-marqo's verson is independent from Marqo.  

- Then, use the following format for releasing changes:
  - . # Release x.y.z 
    - small blurb about release
  - . ## Breaking changes
  - . ## Deprecations
  - . ## Caveats 
    - adding/bumping of dependencies
    - lack of platform support (for certain features)
    - other (non-breaking) caveats to new features/fixes 
  - . ## New features
    - Non-breaking changes
  - . ##  Bug fixes 
  - . ### Testing

### Testing

WARNING: Be very careful if your environment has access to a Marqo production instance. These tests will arbitrarily modify, create and delete indices on the Marqo instances it uses for testing. 


__Tips__

If you don't have the exact python version specified in the `tox.ini` file, you can run tox like this: `tox -e py`. This tells `tox` to use whatever python version it finds on your machine.
#### Unit tests

1. Ensure you have marqo-os running:
```bash
docker rm -f marqo-os
docker run -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" --name marqo-os marqoai/marqo-os:0.0.2-arm
```
2. run `tox` in the Marqo home directory

#### Integration  tests
To run integration tests, pull the [api testing repo](https://github.com/marqo-ai/marqo-api-tests). 
Then, follow the instructions in the [README](https://github.com/marqo-ai/marqo-api-tests#readme).


#### Testing Marqo python client
The Python Marqo client has its own test suite. Clone [the repo](https://github.com/marqo-ai/py-marqo), `cd` into the client home directory and then run `tox`.


