import os
import pathlib
import numpy as np
import cv2
from progressbar import ProgressBar
pbar = ProgressBar()
## Code will be initally based off of David Rowenhorst's (US NRL) 2018 Github code: https://github.com/USNavalResearchLaboratory/NLPAR/)
## Ended up deviating from that quite a bit and optimized the process for numpy. 

# This segment will take a up2 file, load it into an array, and splice that into individual images for edit/export

up2_file_path = pathlib.Path('D:\S3_Heat_Treated_Scans')
sep_image_path = pathlib.Path('F:\Autoencoder Code for Upload\P2Out')
up2_file_name = 'HeatTreat_S3_90deg_firsttrip_600x_20kV_13nA.up2'

#load header
up2_array = np.fromfile(os.path.join(up2_file_path, up2_file_name),dtype=np.uint16, count=30,sep='', offset=0)


#print(up2_array[0],up2_array[1],up2_array[2],up2_array[3],up2_array[4],up2_array[5],up2_array[6],up2_array[7])
#Header is as follows, worth doing a sanity check on this if it breaks, but interpretting the file as 32-bit depth:
#First byte describes the version (1-3, apparently 3 most current but EDAX doesn't export consistently based on settings, closed source piece of shit)
#Version 3 files have additional flags in the header, version 2 and 1 only have 4 values. Here is the kicker, the header is of variable bit depth when
#accounting for those additional flags, but we can just parse the images with the first 4 values, all of which are of 32-bit int depth.For the purpose of parsing and saving
#images this should not matter so long the header remains valid. This is an attempt to cover all the bases. Simple solution to avoid all of this is to
#just never scan using a hexagonal setting

if up2_array[0] == 1:
    version = 1
    pattern_width = up2_array[2]
    pattern_height = up2_array[4]
    data_start = (up2_array[6]/2)
elif up2_array[0] == 2:
    version = 2
    pattern_width = up2_array[2]
    pattern_height = up2_array[4]
    data_start = (up2_array[6]/2)
else:
    version = 3
    pattern_width = up2_array[2]
    pattern_height = up2_array[4]
    data_start = (up2_array[6]/2)


up2_array = np.fromfile(os.path.join(up2_file_path, up2_file_name),dtype=np.uint16, count=-1,sep='', offset=2*int(data_start))
#print(up2_array[0],up2_array[1],up2_array[2],up2_array[3],up2_array[4],up2_array[5],up2_array[6],up2_array[7],up2_array[8],up2_array[9],up2_array[10],up2_array[11],up2_array[12],up2_array[13],up2_array[14],up2_array[15],up2_array[16],up2_array[17],up2_array[18],up2_array[19], up2_array[21],up2_array[22],up2_array[23],up2_array[24],up2_array[25],up2_array[26],up2_array[27], up2_array[28])
#assumes data is parsed in 16-bit unsigned (i.e. up2), if not (i.e. 8-bit up1) you'll need to tinker with this a bit to calculate the correct number of patterns
#but thats all this code does.

fileSize = os.stat(os.path.join(up2_file_path, up2_file_name))
print(fileSize.st_size)
patternSize = pattern_width*pattern_height*2 #16 bits means times 2
num_patterns = int(fileSize.st_size/patternSize)
print(num_patterns)

#Parse massed array (essentially 1 big list of every byte) and reshape into image dimensions and a parsable array because 16-bit dtyping seems to break everything.
up2_array = np.reshape(up2_array,(pattern_width,pattern_height,num_patterns), order='F')
print(up2_array.shape)

#This takes that array and splices it into jpgs (much faster than Fiji)

def export_up2_jpg(num_patterns):
    for i in pbar(range(num_patterns)):
        fname = str(i)
        img = up2_array[:,:,i]
        
        #Transposing the image is neccesary due to np array conventions
        
        img = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
        img = cv2.flip(img,1)
        img = img / 255
        cv2.imwrite(os.path.join(sep_image_path, fname+str('.jpg')), img)


export_up2_jpg(num_patterns)