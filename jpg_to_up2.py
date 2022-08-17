import os
import pathlib
import numpy as np
import cv2
from progressbar import ProgressBar
pbar = ProgressBar()

#Where are you saving the UP2 file, what is its name, and where are the images coming from? Do that here.
up2_file_path = pathlib.Path('F:\Autoencoder Code for Upload')
img_data = pathlib.Path('F:\Autoencoder Code for Upload\P2Out')
up2_file_name = 'test_output_for_githhub.up2'

def loose_2_up2(img_data):
    #Instantiate np array so python quits yelling.
    img_container = []
    #img_container = np.asarray(img_container)
    print(len(sorted(os.listdir(img_data))))
    for i in pbar(range(len(sorted(os.listdir(img_data))))):
        #print(i)
        f_name = str(i)+'.jpg'
        img = os.path.join(img_data,f_name)    
        img = cv2.imread(img)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #img = img * 255
        img = np.asarray(img)
        
        #print(img)
        #Desired resolution goes here. Not really critical and could be commented out if your input images are of the correct resolution.
        #TSL OIM seems indifferent so long this matches what is in the header, so this is here to catch that error.
        #See the header information below and make sure to change it if you comment this out.
        img = cv2.resize(img, (235,235))
        img = img.flatten()
        #print(img.shape)
        img = img.astype(np.uint16)
        img_container.append(img)
        
    img_container = np.asarray(img_container)
    img_container.flatten()
    return img_container
    
def flatten_loose_array(img_array):
    flat_list = []
    for element in img_array:
        for item in element:
            flat_list.append(item)
    return flat_list

img_array = loose_2_up2(img_data)
img_array = img_array * 255
img_array.astype(np.uint16)
#img_array = int(img_array)
#img_array.reshape(img_array.size)
#dummy header
#header = up2_array_header
#This is a 'dummy' header used to repackage the images into a readable UP2 file. The 235 is the pattern resolution. Change to match your
#images or resize your images to match, but make sure its consistent otherwise your pattern file will be jumbled.
header = [1,0.0,235,0.0,235,0.0,16,0.0]
header = np.asarray(header)
header.astype(np.uint16)
img_array = np.insert(img_array, [0], header)
#img_array = np.append(header,img_array)
#img_array=bytearray(img_array)
#print(np.shape(img_array))
print(img_array[0],img_array[1],img_array[2],img_array[3],img_array[4],img_array[5],img_array[6],img_array[7],img_array[8])
img_array.tofile(os.path.join(up2_file_path, up2_file_name))