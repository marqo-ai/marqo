import uuid
import os
from yt_dlp import YoutubeDL
from pydub import AudioSegment
import urllib.request
import shutil
from typing import List
import hashlib
from multiprocessing import Pool

ABS_FILE_FOLDER = os.path.dirname(os.path.abspath(__file__))


class AudioWrangler:
    def __init__(self, output_path: str, clean_up: bool = True):

        self.output_path = output_path

        self.tmp_dir = "downloads"

        os.makedirs(os.path.join(ABS_FILE_FOLDER, self.tmp_dir), exist_ok=True)

        if clean_up:
            self.clean_up()

    def download_from_file(self, file):
        urls = []
        with open(file, "r") as f:
            for url in f.readlines():
                urls.append(url.strip())
        self.multiprocess_read_url_sources(urls)

    def multiprocess_read_url_sources(self, sources: List[str]):
        pool = Pool(os.cpu_count())
        pool.map(self.read_url_source, sources)

    def read_url_source(self, source: str):
        if "www.youtube.com" in source:
            return self.download_from_youtube(source)

        return self.download_from_web(source)

    def clean_up(self):
        for file in os.listdir(os.path.join(ABS_FILE_FOLDER, self.tmp_dir)):
            os.remove(os.path.join(ABS_FILE_FOLDER, self.tmp_dir, file))

    def download_from_youtube(self, url: str):
        outf = os.path.join(
            ABS_FILE_FOLDER,
            self.tmp_dir,
            hashlib.sha256(url.encode("ascii")).hexdigest(),
        )
        ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
            "fragment_retries": 10,
            "outtmpl": outf,
        }
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        outf = self.convert_to_wav(outf + ".mp3")
        outf = self._move_to_output(outf)
        return outf

    def download_from_web(self, url: str):
        outf = os.path.join(
            ABS_FILE_FOLDER,
            self.tmp_dir,
            hashlib.sha256(url.encode("ascii")).hexdigest() + f".wav",
        )
        req = urllib.request.Request(url=url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response, open(outf, "wb") as out_file:
            shutil.copyfileobj(response, out_file)
        outf = self.convert_to_wav(outf)
        outf = self._move_to_output(outf)
        return outf

    def convert_to_wav(self, fpath: str):
        sound = AudioSegment.from_file(fpath)
        wav_path = "".join([p for p in fpath.split(".")[:-1]]) + ".wav"
        sound.export(wav_path, format="wav")
        return wav_path

    def _move_to_output(self, file):
        target = os.path.join(self.output_path, os.path.basename(file))
        shutil.move(file, target)
        return target
