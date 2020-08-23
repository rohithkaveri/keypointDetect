import numpy as np
import cv2
import math
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import matplotlib.pyplot as plt
from GPSPhoto import gpsphoto



def get_exif(filename):
    exif = Image.open(filename)._getexif()

    if exif is not None:
        for key, value in exif.items():
            name = TAGS.get(key, key)
            exif[name] = exif.pop(key)

        if 'GPSInfo' in exif:
            for key in exif['GPSInfo'].keys():
                name = GPSTAGS.get(key,key)
                exif['GPSInfo'][name] = exif['GPSInfo'].pop(key)

    return exif


def get_decimal_coordinates(info):
    for key in ['Latitude', 'Longitude']:
        if 'GPS'+key in info and 'GPS'+key+'Ref' in info:
            e = info['GPS'+key]
            ref = info['GPS'+key+'Ref']
            info[key] = ( e[0][0]/e[0][1] +
                          e[1][0]/e[1][1] / 60 +
                          e[2][0]/e[2][1] / 3600
                        ) * (-1 if ref in ['S','W'] else 1)

    if 'Latitude' in info and 'Longitude' in info:
        return [info['Latitude'], info['Longitude']]


#

img = cv2.imread('/Users/rohithkaveri/real_img.jpg', 0)

# Initiate STAR detector
orb = cv2.ORB_create()

# find the keypoints with ORB
kp = orb.detect(img, None)

print(len(kp))
totalKP = len(kp)

# compute the descriptors with ORB
kp, des = orb.compute(img, kp)

# draw only keypoints location,not size and orientation
img2 = cv2.drawKeypoints(img, kp, outImage=None, color=(0, 255, 0), flags=0)
plt.imshow(img2), plt.show()

imgCopy = img.copy()
imgCopyPIL = Image.open('/Users/rohithkaveri/real_img.jpg')
startPtx = int(0)
startPty = int(0)
toIncrement = int(100)
density = 0
densitySearch = int(0.05*totalKP)  # looking for 5% of keypoints in any box 100x100
maxDensityBoxes0 = []  # for the first (x,y) point for a rectangle
maxDensityBoxes1 = []  # for the second (x,y) point for a rectangle, 9x6 for 1773x1182

height = int(math.floor(imgCopy.shape[0]))
#print(height)
length = int(math.floor(imgCopy.shape[1]))
#print(length)

# add conditions for left edge, right edge, bottom and top edges if statements

for i in range(0, int(height/toIncrement)):
    startPtx = int(0)
    y0 = startPty
    y1 = startPty + toIncrement
    for j in range(0, int(length/toIncrement)):
        density = 0
        x0 = startPtx
        x1 = startPtx + toIncrement
        startSetx = (x0, y0)
        box = cv2.rectangle(imgCopy, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 2)
        #cv2.imshow('imgbox', imgCopy)
        #cv2.waitKey(0)
        for k, keypoint in enumerate(kp):
            if x0 <= keypoint.pt[0] <= x1 and y0 <= keypoint.pt[1] <= y1:
                density = density+1
        if density >= densitySearch:
            # print(density)
            # print((x0, y0))
            # print((x1, y1))
            maxDensityBoxes0.append((x0, y0))
            maxDensityBoxes1.append((x1, y1))
        startPtx = startPtx + 100
    startPty = startPty + 100

croppedImgList = []
for i, pts in enumerate(maxDensityBoxes0):
    print(maxDensityBoxes0[i])
    if (maxDensityBoxes0[i][0] > 0 and maxDensityBoxes0[i][1] > 0) and (maxDensityBoxes1[i][0] > 0 and
                                                                        maxDensityBoxes1[i][1] > 0):
        newImg0 = imgCopyPIL.crop((maxDensityBoxes0[i][0]-50, maxDensityBoxes0[i][1]-50, maxDensityBoxes1[i][0]+50,
                                  maxDensityBoxes1[i][1]+50))
        newImg = newImg0.resize((350, 350))
        newImg.show('croppedimg', newImg)
        croppedImgList.append(newImg)
    else:
        newImg0 = imgCopyPIL.crop((maxDensityBoxes0[i][0], maxDensityBoxes0[i][1], maxDensityBoxes1[i][0],
                                  maxDensityBoxes1[i][1]))
        newImg = newImg0.resize((350, 350))
        newImg.show('croppedimg', newImg)
        croppedImgList.append(newImg)

path = '/Users/rohithkaveri/Desktop/'
for j, imgs in enumerate(croppedImgList):
    #cv2.imwrite(os.path.join(croppedImgList[j]))
    croppedImgList[j].save(path+'crop_'+str(j)+'.jpg', 'JPEG', optimize=True)

# Get the data from image file and return a dictionary
data = gpsphoto.getGPSData('/Users/rohithkaveri/real_img.jpg')
#print(data['Latitude'], data['Longitude'])
