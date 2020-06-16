import xml.etree.ElementTree as ET

from . import np
from . import pd
from . import cv2

from glob import glob
from os import path as pathlib

# COLS = ['filename','path','width','height','class','xmin','ymin','xmax','ymax']


class JSON(object):
    """
    Helper Class For Mapping JSON vars to Objects
    """
    def __init__(self,data=dict(),inner=False):
        for key in data:
            if type(data[key]) == dict:
                self.__dict__[key] = JSON(data[key],inner=True)
            else:
                self.__dict__[key] = data[key]

    def __repr__(self):
        return self.__dict__.__str__()
    
    def __str__(self):
        return self.__dict__.__str__()
    
    def __getitem__(self,key):
        return self.__dict__[key]
    
    def __setitem__(self,key,value):
        self.__dict__[key] = value
    
    def __iter__(self):
        for key in self.__dict__:
            if type(self.__dict__[key]) == JSON:
                yield key, self.__dict__[key]()
            else:
                yield key, self.__dict__[key]
        
    def __call__(self,):
        return {i:j for i,j in  self.__iter__()}

class XMLParser(object):
    """
    XML parser/helper 
    """
    def __init__(
            self,
            path:str,
        ):
        self.path = pathlib.abspath(path)
        self.data = []

    def __getitem__(self,key):
        return self.data[key]
    
    def __len__(self,):
        return len(self.data)

    def __repr__(self):
        return f"XMLParser @ {self.path}"

    def parse(self,):
        data = []
        for p in glob(pathlib.abspath(f"{self.path}/annotations/*.xml")):
            try:
                tree = ET.parse(p)
            except:
                continue
            row = JSON()
            row.filename = tree.find("filename").text
            row.path = pathlib.abspath(f"{self.path}/images/{row.filename}")
            row.height = int(tree.find("size").find("height").text)
            row.width = int(tree.find("size").find("width").text)
            boxes = []
            for o in tree.findall("object"):
                box = JSON()
                box.category = o.find("name").text
                for c in ['xmin','ymin','xmax','ymax']:
                    box[c] = int(o.find("bndbox").find(c).text)
                
                box.h = (box.ymax - box.ymin) / row.height
                box.w = (box.xmax - box.xmin) / row.width

                box.y = (box.ymin / row.height ) + (box.h / 2 ) 
                box.x = (box.xmin / row.width ) + (box.w / 2 ) 

                boxes.append(box)
            row['boxes'] = boxes
            data.append(row)
        self.data = data

    def as_dataframe(self,):
        rows = []
        for row in self.data.copy():
            for box in row.boxes:
                box.path = row.path
                box.filename = row.filename
                box.width = row.width
                box.height = row.height
                rows.append(box)
        
        return pd.DataFrame([r() for r in rows])


def read_images(data:XMLParser,img_size:int):
    """
    Returns batch of images with ( img_size x img_size x 3 ) size ang RGB channel format.
    """
    return np.array([
        cv2.cvtColor(
            cv2.resize(
                cv2.imread(d.path),
                (img_size,img_size)
            ),
            cv2.COLOR_BGR2RGB
        )
        for 
            d
        in 
            data
    ])