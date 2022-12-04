import gdown


def download_from_drive(url: str, destination: str):
    gdown.download(url, destination, quiet=False, fuzzy=True)
