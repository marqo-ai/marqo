import os
from SpeechSearch.downloader import AudioWrangler


def main():
    os.makedirs("audios", exist_ok=True)
    print("Downloading...")
    aw = AudioWrangler("audios")
    aw.download_from_file("coffee_links.txt")
    aw.download_from_file("podcast_links.txt")
    aw.download_from_file("meeting_links.txt")
    print("Done!")


if __name__ == "__main__":
    main()
