import  os
import  cv2
import  random
dir = "../datasets/rawDogData"
newDir = "../datasets/dogData/Images"

# We  created  a new  directory  for  selected  dataset.
idx = 0

for  dogType  in os.listdir(dir):
    dogDir = os.path.join(dir ,dogType)
    #  Enters  to  breed  types
    for  dogPhoto  in os.listdir(dogDir):
        img = cv2.imread(os.path.join(dogDir ,dogPhoto))
        img = cv2.resize(img ,(256 ,256))
        imgflipped = cv2.flip(img, 1)

        #  Resizing  every  image to  same  size
        writeName = 'dogPhotoFlipped'+str(idx)+'.jpg'

        cv2.imwrite(os.path.join(newDir ,dogPhoto), img)
        cv2.imwrite(os.path.join(newDir ,writeName), imgflipped)
        idx += 1
