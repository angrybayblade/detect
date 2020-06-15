from requests import get
from tqdm.cli import tqdm
from os import system
from math import ceil

URL = "https://raw.githubusercontent.com/Shenggan/BCCD_Dataset/master/BCCD"
CHUNK_SIZE = 1024

def main():
    print ("[+] Downloading Annotations")
    for i in tqdm(range(1,411)):
        i = str(i)
        file = f"BloodImage_{(5-len(i))*'0'}{i}"
        url = f"{URL}/Annotations/{file}.xml"
        req = get(url)
        with open(f"Annotations/{file}.xml","wb") as xml:
            xml.write(req.content)

    print ("[+] Downloading Images")
    for i in tqdm(range(1,411)):
        i = str(i)
        file = f"BloodImage_{(5-len(i))*'0'}{i}"
        url = f"{URL}/JPEGImages/{file}.jpg"
        req = get(url)
        with open(f"JPEGImages/{file}.jpg","wb") as jpg:
            jpg.write(req.content)

if __name__ == "__main__":
    main()
