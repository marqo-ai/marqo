import marqo
import soundfile as sf
import os
import librosa
import numpy as np
import json

from SpeechSearch.searcher import search_transcripts


def get_audio(audio_path: str, samplerate: int) -> np.ndarray:
    audio, _ = librosa.load(audio_path, sr=samplerate)
    return audio


def main():
    mq = marqo.Client(url="http://localhost:8882")

    index_name = "transcription-index"
    while True:
        query = input("Enter a query: ")
        results, segments = search_transcripts(
            query=query, limit=2, index=index_name, mq=mq, audio_getter=get_audio
        )

        print(json.dumps(results, indent=4))

        fname = "-".join(
            ["".join([c for c in w if c.isalnum()]) for w in query.split()]
        )

        print(f"Results written to {fname} (txt and wav)!")

        for idx, s in enumerate(segments):

            with open(os.path.join("results", f"{fname}_{idx}.txt"), "w+") as f:
                f.write(s["transcription"])

            sf.write(
                os.path.join("results", f"{fname}_{idx}.wav"),
                s["audio"],
                s["samplerate"],
            )

        print("Done!")


if __name__ == "__main__":
    main()
