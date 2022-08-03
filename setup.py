from setuptools import setup, find_packages
# from src.marqo.version import __version__

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

setup(
    install_requires=[
        # difficult packages:
        "click==8.0.4",
        # client:
        "requests",
        "urllib3",
        # s2_inference:
        "clip-marqo==1.0.1",
        "more_itertools",
        "nltk",
        "torch",
        "pillow",
        "numpy",
        "validators",
        "sentence-transformers",
        "onnxruntime",
        "onnx",
        "protobuf==3.20.1",
    ],
    name="marqo",
    version="0.1.10",
    author="marqo org",
    author_email="org@marqo.io",
    description="Neural search for humans",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src", exclude=("tests*",)),
    keywords="search python marqo opensearch neural semantic vector embedding",
    platform="any",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    python_requires=">=3",
    package_dir={"": "src"},
)
