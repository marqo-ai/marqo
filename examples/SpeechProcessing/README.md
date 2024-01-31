# Search Speech Data

Original article [here](https://www.marqo.ai/blog/speech-processing).

## Overview

This is a small project that indexes the transcriptions of audio files into Marqo to create a searchable database which can be queried with text and return corresponding segments of audio.

### Examples
After downloading and indexing the data, the system can answer domain specific questions about a range of topics (using `3b. Chat.py`):

```
Q: What pressure should the machine use to extract espresso?

A: The pressure of the machine is an important factor when it comes to extracting espresso. Ideally, the machine should be set to 9 bars of pressure in order to get a strong and concentrated espresso with a thick and sweet flavour. A lower pressure may lead to a slower shot and a longer contact time, but it won't necessarily increase the extraction. Brewing too fast can lead to channeling, which reduces extraction and causes a harsh and sour taste.
```

```
Q: What is the controversy around Samsungs Space Zoom feature?

A: The controversy around Samsung's Space Zoom feature is that it implies that their phone cameras can take high definition pictures of the moon and other distant objects, whereas in reality the feature is only able to zoom in on far away objects with a low degree of clarity. This has raised concerns about the use of this technology for deep fake videos, as well as worries that Samsung may be purposely misrepresenting the capability of their phones.
```

```
Q: What are some application of GANs that are beneficial to the research community?

A: GANs have seen a lot of applications in recent years that are beneficial to the research community. GANs have been used to generate realistic images, improve security, generate music and more. GANs are especially helpful to students or practitioners in research fields as they can greatly help with the problems they see in their day-to-day. GANs are also being used for good in areas such as differential privacy and combating bias. GANs are being used in the medical field as well, where they can generate fake medical data for use in research without compromising any patient's privacy. GANs are also being used for fairness and for creating data for underrepresented populations. GANs have been used to learn from unlabeled real images and make a refiner that upgrades the synthetic image to make it more realistic.
```


## Getting Started

1. Install requirements:

    ```
    python -m venv venv
    ```

    Linux/Mac
    ```
    source venv/bin/activate
    ```
    Windows
    ```
    venv\Scripts\activate
    ```

    ```
    pip install -r requirements.txt
    ```

2. Install and run Marqo:

    [See the getting start guide for Marqo](https://github.com/marqo-ai/marqo#Getting-started)

3. Install FFmpeg for your platform.
4. [Log in to Hugging Face to get an API key for pyannot speaker-diarization](https://huggingface.co/pyannote/speaker-diarization)
5. [Get an OpenAI API Key](https://platform.openai.com/account/api-keys).
6. Copy `.env_local` and rename it as `.env`, place the API keys in the `.env` file

## Troubleshooting

### Symbolic link creation related errors in `2. Process.py`

PyAnnote may raise these errors if it is not running with the appropriate privileges, run the script as admin/super user to fix.

## Usage

Download the data:

```
python '1. Download.py'
```

Process and index the data:

(for this step it is highly recommended that you have a GPU)

```
python '2. Process.py'
```

Search the data (this gives you an interactive prompt and returns direct excerpts):

```
python '3a. Search.py'
```

Question answer the data (this gives you an interactive prompt and returns natural language responses):

```
python '3b. chat.py'
```
