from requests import get
from tqdm.cli import tqdm
from os import system
from math import ceil

URL = "https://arcraftimages.s3-accelerate.amazonaws.com/Datasets/Ships/ShipsPascalVOC.zip?region=us-east-22"
CHUNK_SIZE = 1024

def main():
    req = get(URL,stream=True)
    with open("data.zip","wb") as file:
        for chunk in tqdm(req.iter_content(chunk_size=CHUNK_SIZE),total=ceil(int(req.headers['content-length'])/CHUNK_SIZE)):
            file.write(chunk)

    system("unzip data.zip")
    system("rm data.zip")
    system("rm -rf __MACOSX")

if __name__ == "__main__":
    main()
