Thank you for your interest in making neural search more accessible!

---

## Ways to contribute

### Bug reports
You can submit bug reports on our GitHub repo

### Contributing code
We welcome contributions to the codebase. 

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