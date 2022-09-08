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

### Releasing changes
- Generate a pull request to main. These will be reviewed before merging
- To release a change, bump the correct version (major.minor.patch, see below for more details), run all the tests, build the package and push to it PyPi
- We do regular minor releases. These are releases that are additive and don't break existing functionality. During a minor release we bump the minor version number: 0.41.24 -> 0.42.24
- In-between minor releases, we can release bug-fixes by bumping the patch number: 0.34.6 -> 0.34.7
- If there are any breaking changes, there will be a major release: 5.39.55 -> 6.39.55
- Not that pre-version 1.x.x, because this is the pre-release stage, breaking functionality may be introduced in minor versions.

### Testing
Notes: 

- If you don't have the exact python version specified in the `tox.ini` file, you can run tox like this: `tox -e py`. This tells `tox` to use whatever python version it finds on your machine. 
- Be very careful if your environment has access to a Marqo production instance. These tests will arbitrarily modify, create and delete indices on the Marqo instances it uses for testing. 

#### Unit tests
To run unit tests, simple run `tox` in the Marqo home directory

#### Integration  tests
To run integration tests, pull the [api testing repo](https://github.com/marqo-ai/marqo-api-tests). 
Then, follow the instructions in the [README](https://github.com/marqo-ai/marqo-api-tests#readme).


#### Testing Marqo python client
The Python Marqo client has its own test suite. Clone [the repo](https://github.com/marqo-ai/py-marqo), `cd` into the client home directory and then run `tox`.


