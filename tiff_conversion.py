import sys
import gpudb
from io import StringIO
import collections
from avro import schema, io
import os
from PIL import Image

# from io import StringIO

# Init
table_name = "imageTableInference"
collection = "MASTER"
tiffFolder = "./imagesTIF/"
jpegFolder = "./imagesJPEG/"
# init
gpudb_host = "172.31.33.26"
h_db = gpudb.GPUdb(encoding='BINARY', host=gpudb_host, port='9191')
my_type = """
{
   "type": "record",
   "name": "image",
   "fields": [
   {"name":"image","type":"bytes"}
   ]
}""".replace('\n', '').replace(' ', '')

def creatTable(type_def=my_type, table_name="imageTableInference", collection="MASTER"):
    response = h_db.create_type(type_definition=type_def,label='image')
    type_id = response['type_id']
    if h_db.has_table(table_name=table_name):
        h_db.clear_table(table_name=table_name)
    response = h_db.create_table(table_name=table_name,type_id=type_id,options={"collection_name": collection})

# Convert tiff images to jpeg
def Tiff2Jpeg(imInFolder="./imagesTIF/", imOutFolder="./imagesJPEG/"):
    listImages = os.listdir(imInFolder)
    for image in listImages:
        if image[-4:] == "tiff":
            im = Image.open(imInFolder + image)
            im.save(imOutFolder + image[:-4] + "png")

# Load the image to database
def ingestImage(imageFolder="./imagesJPEG/", tableName="imageTableInference", collection="MASTER"):
    datum = collections.OrderedDict()
    encoded_obj_list = []
    listImages = os.listdir(imageFolder)
    for image in listImages:
        myimage = open(imageFolder + image, "r").read()
        datum["image"] = myimage
        encoded_obj_list.append(h_db.encode_datum(my_type, datum))
    # print encoded_obj_list
    response = h_db.insert_records(table_name=tableName,data=encoded_obj_list,list_encoding='binary',options={})
    print(response)

if __name__=="__main__":
    creatTable()
    Tiff2Jpeg(tiffFolder, jpegFolder)
    ingestImage(jpegFolder)

