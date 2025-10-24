Steps to start running backwards compatibility tests on your local machine:

1. If starting for the first time, create a virtual environment and install the required packages:
2. Activate the virtual environment
3. Make sure the docker daemon is running 
4. Configure your AWS credentials
5. Add marqo repo to your Python path: `cd marqo` then `export PYTHONPATH="$PWD/src:$PWD:$PYTHONPATH"`
6. python3 tests/compatibility_tests/compatibility_test_runner.py  --mode "backwards_compatibility"   --from_version "2.10.2"   --to_version "2.14.1"   --to_image "424082663841.dkr.ecr.us-east-1.amazonaws.com/marqo-compatibility-tests@sha256:c1c596f900e10b48e1ea6ff66e22f4d2da3d5b684fc08b02e3ad11baa21f9294"
7. For rollback just pass the mode as rollback
8. All the folders here are API folders and the tests are written in the respective API folders corresponding to the API they test.
9. When developing, if you create a subfolder inside an API folder, make sure to declare it as a package by creating an empty __init__.py file inside the subfolder.
10. When writing tests, make sure to create an index with a unique name (defined in settings dict), such that it does not conflict with other tests that are running at the same time.
11. Every test case should be in a new .py file.
12. Always include all indexes in the test in cls.indexes_to_test_on so they can be cleaned up.