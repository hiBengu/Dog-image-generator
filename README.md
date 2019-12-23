# Generating Dog Images with DCGAN

This project creates artificial dog images with a basic DCGAN implementation. 

This is an assignment given in "Machine Learning for Signal Processing" course in Istanbul Technical University.

"dataTransforms.py" reads unprocessed images, by flipping and cropping them creates a triple sized dataset and saves them. 

"dogMain.py" is the main file which is needed to train the model. Inside the dogMain, following scripts are called.
  "dogModel.py" contains generator and discriminator network classes.
  "dogDataLoader.py" reads the dataset and save to use them later.
  "dogTrainer.py" returns a class for training. Loss and optimization operations are inside this script
  "dogFinalize.py" saves the model, and figures at the end of the training.
  
"evalDog.py" loads the trained generator network and creates new artificial dog images from noise.

"viewFakeDogs.py" is a script to analyze the the iamges that saved through the training.
