from setuptools import setup, find_packages
# from src.marqo.version import __version__

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

setup(
    install_requires=[
        # difficult packages:
        "click==8.0.4",
        # tensor_search:
        "requests",
        "urllib3",
        "fastapi_utils",
        # s2_inference:
        "clip-marqo==1.0.2",
        "more_itertools",
        "nltk",
        "torch",
        "open_clip_torch",
        "pillow",
        "numpy",
        "validators",
        "sentence-transformers",
        "onnxruntime-gpu",
        "onnx",
        "protobuf==3.20.1",
        "pandas",
        "optimum",
        "fastapi",
        "uvicorn[standard]",
        "fastapi_utils",
        "opencv-python-headless",
        "psutil",
        "multilingual_clip"
    ],
    name="marqo-engine",
    version="0.1.10",
    author="marqo org",
    author_email="org@marqo.io",
    description="Tensor search for humans",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src", exclude=("tests*",)),
    keywords="search python marqo opensearch vespa neural tensor semantic vector embedding",
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