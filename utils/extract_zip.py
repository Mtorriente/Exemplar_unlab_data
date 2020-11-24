#Extraemos zip subconjunto de STL1O o ISIC
from zipfile import ZipFile
#fichero = "Subconjunto_ISIC_10.zip"
fichero = "classes_folder_10.zip"
with ZipFile(fichero,'r') as zip:
  zip.extractall()
  print("Done extracting file")
