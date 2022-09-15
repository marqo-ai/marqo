# Marqo Streamlit Demo Application

### Prerequisites

What things you need to install the software and how to install them.

```
Python 3.8
```

### Getting Started

1. Download the Dataset from
    [Clothing Dataset](https://github.com/alexeygrigorev/clothing-dataset) into the directory where the `streamlit_marqo_demo.py` script is found.

2. Run this command inside the script directory to setup an HTTP server
    ```
    python3 -m http.server 8222
    ```
    This is for the marqo docker container to read files from local os.
    For more info on this please visit [this link](https://github.com/marqo-ai/marqo/issues/35).

3. Make sure to run the Marqo docker container via the following command:
    ```
    docker run --name marqo -it --privileged -p 8882:8882 --add-host host.docker.internal:host-gateway marqoai/marqo:0.0.3
    ```

4. Install Streamlit. This can be done by via the following [link](https://docs.streamlit.io/library/get-started/installation).

5. Install Marqo
    ```
    pip install marqo
    ```
    Note: if you are using Anaconda, make sure to install marqo in the anaconda environment.

6. Once Streamlit is installed, we can start the Streamlit application by running the following command inside the directory where the `streamlit_marqo_demo.py` script is located:
    ```
    streamlit run streamlit_marqo_demo.py
    ```

For more information on: 
- Marqo's functions and features, please visit the [Marqo Documentation Page](https://marqo.pages.dev/).

- Streamlit's functions and features, please visit the [Streamlit Documentation Page](https://docs.streamlit.io/).

## Usage
Feel free to checkout the code in order to have a better understanding on how Marqo functions are used :).
