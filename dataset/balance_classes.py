"""
This file takes 6000 positive and 6000 negative instances from the training folder and moves it in the subset_training
folder.
If you need to move class1, change the last line to getclass1, else leave getclass0.
you need to specify your own paths. (origin, destination)
"""
import os
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import shutil
def getclass1(source,path): #takes the images of class1 and copies into another folder.
    files= os.listdir(source) #after this I will take the same number of classes of zero and copy to the same folder.
    print(files)
    i=0
    os.chdir(source)
    for file in files:   
        file_name=os.path.splitext(os.path.basename(file))[0]
        label=int(file_name.split('_')[4][5])
        if label==1:
            img=Image.open(file)
            img.save(path+file,format='PNG')
            i+=1
def getclass0(source,path): #takes the images of class1 and copies into another folder. This is done to know how many classes of 1 
    files= os.listdir(source) 
    i=0
    count=0
    os.chdir(source)
    for file in files: 
        print(file)
        if count!=6000:
            file_name=os.path.splitext(os.path.basename(file))[0]
            label=int(file_name.split('_')[4][5])
            if label==1:
                shutil.move(source+file,path)
                i+=1
                count+=1

if __name__=="__main__": 
    origin='/home/ubuntu/preprocessing/maincode/files/training/'
    destination='/home/ubuntu/preprocessing/maincode/files/subset_training/'
    getclass0(origin, destination)