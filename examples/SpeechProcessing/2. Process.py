import marqo
import os
import torch
from dotenv import load_dotenv
import time
from SpeechSearch.transcriber import AudioTranscriber
from SpeechSearch.indexer import index_transciptions

load_dotenv()


def check_can_run() -> None:
    """Helper to check if there are any files to process"""
    if not os.path.exists("audios"):
        raise FileExistsError("There is no ./audios/ folder for input files!")
    os.makedirs("results", exist_ok=True)


def main():

    marqo_device = "cpu"

    check_can_run()

    mq = marqo.Client(url="http://localhost:8882")
    index_name = "transcription-index"

    mq.delete_index(index_name)
    settings = {
        "index_defaults": {
            "text_preprocessing": {
                "split_length": 2,
                "split_overlap": 0,
                "split_method": "sentence",
            },
        },
    }

    response = mq.create_index(index_name, settings_dict=settings)

    if torch.cuda.is_available():
        local_device = "cuda"
    else:
        local_device = "cpu"

    print(f"Transcribing using {local_device} and indexing using {marqo_device}")

    print("NOTE: Transcribing and indexing may take some time...")

    if local_device == "cpu":
        print(
            "WARNING: You are using a CPU for speaker annotation and transcription. This can be very slow - it is recommended that you only attempt to use a small number of short audio files!"
        )
        print("Using the audio files downloaded in step 1 may take hours on CPU")

    at = AudioTranscriber(os.environ["HF_TOKEN"], local_device)

    for idx, f in enumerate(os.listdir("audios")):
        print(f"Processing file {idx+1} of {len(os.listdir('audios'))}...")

        at_start = time.perf_counter()
        if ".wav" not in f:
            continue
        file = os.path.join("audios", f)

        annotated_transcriptions = at.process_audio(file)
        at_end = time.perf_counter()

        print(
            f"Annotation and transcription of file {idx+1} took {at_end-at_start} seconds"
        )

        indx_start = time.perf_counter()
        index_transciptions(
            annotated_transcriptions,
            index_name,
            mq,
            ["samplerate", "start", "end", "speaker", "file"],
            marqo_device,
        )
        indx_end = time.perf_counter()

        print(f"Indexing of file {idx+1} took {indx_end-indx_start} seconds")

    print("Done!")


if __name__ == "__main__":
    main()
