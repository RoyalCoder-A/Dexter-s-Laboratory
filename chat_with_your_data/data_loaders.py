from pathlib import Path
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import OpenAIWhisperParser
from langchain_community.document_loaders import YoutubeAudioLoader

if __name__ == "__main__":
    dir = Path(__file__).parent / "data"
    loader = PyPDFDirectoryLoader(dir / "tmp")
    docs = loader.load()
    print(docs[1])
    youtube_loader = GenericLoader(
        YoutubeAudioLoader(["https://www.youtube.com/watch?v=ffqMZ5IcmSY"], str(dir)),
        OpenAIWhisperParser(),
    )
    youtube_data = youtube_loader.load()
    print(youtube_data[0])
