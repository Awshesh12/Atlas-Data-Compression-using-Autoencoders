# Atlas-Data-Compression-using-Autoencoders
The given repository uses deep autoencoders for compressing data of ATLAS large hadron collider from 4 variables to 3 variables.

## Dataset
all_jets_test_4D_100_percent.pkl and all_jets_train_4D_100_percent.pkl are the testing and training datasets or you can say validation and training. 
These datasets are loaded into the notebook by the help of pandas library. You can find it in the Preprocessing.py file.

## Neural net
The neural net is located in the model.py. I have tested a lot of neural nets by varying the activation functions for each of them and have put the most accurate model on the basis of r2_score and mean absolute error.

## Preprocessing of the data
The data has to be normalized and converted to tensor form before passing it through the model. This can be executed by preprocessing.py

## Training and Validation
Model can be trained using Training and validation.py.

# How to run the files 
1) Refer to AutoEncodersAtlasDataCompression2.ipynb for getting a systematic idea how these files are being run together and how the plots are being constructed.
2) If you want to use python files, then start from Preprocessing.py, then we have to go for model.py for building the architecture of the neural network. 
3) The Training and validation.py can be used for training the model and validating it on the testing set.
4) For Plotting the results, different sections of the Plots and Results.py have been differentiated for preventing the confusion. These different sections will plot different graphs.

