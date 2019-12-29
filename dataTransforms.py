import  os
import  cv2
import  random
dir = "/home/itu-1070/hibengu/github/datasets/rawDogData"
newDir = "/home/itu-1070/hibengu/github/datasets/dogData/Images"

# We  created  a new  directory  for  selected  dataset.
idx = 0

for  dogType  in os.listdir(dir):
    dogDir = os.path.join(dir ,dogType)
    #  Enters  to  breed  types
    for  dogPhoto  in os.listdir(dogDir):
        img = cv2.imread(os.path.join(dogDir ,dogPhoto))
        imgOrg = cv2.resize(img ,(256 ,256))
        imgflipped = cv2.flip(img, 1)
        #imgCropped = img[int(img.shape[0]/2)-128:int(img.shape[0]/2)+128, int(img.shape[1]/2)-128:int(img.shape[1]/2)+128]

        # if  (idx % 2 == 0):
        #     M = cv2.getRotationMatrix2D((imgOrg.shape[0]/2,imgOrg.shape[1]/2), 45, 1.0)
        #     imgRotated = cv2.warpAffine(imgOrg, M, (imgOrg.shape[0], imgOrg.shape[1]))
        #     print(imgRotated.shape, " -- ", idx)
        # elif (idx % 2 == 1):
        #     M = cv2.getRotationMatrix2D((imgOrg.shape[0]/2,imgOrg.shape[1]/2), -45, 1.0)
        #     imgRotated = cv2.warpAffine(imgOrg, M, (imgOrg.shape[0], imgOrg.shape[1]))

        #  Resizing  every  image to  same  size
        writeName = 'dogPhoto' + str(idx) + '.jpg'
        writeNameFlipped = 'dogPhotoFlipped'+str(idx)+'.jpg'
        writeNameCropped = 'dogPhotoCropped'+str(idx)+'.jpg'
        # writeNameRotated = 'dogPhotoRotated'+str(idx)+'.jpg'

        cv2.imwrite(os.path.join(newDir ,writeName), imgOrg)
        cv2.imwrite(os.path.join(newDir ,writeNameFlipped), imgflipped)
        # if (imgCropped.shape[0] == 256 and imgCropped.shape[1] == 256):
        #     cv2.imwrite(os.path.join(newDir ,writeNameCropped), imgCropped)
        
        # cv2.imwrite(os.path.join(newDir, writeNameRotated), imgRotated)
        idx += 1
