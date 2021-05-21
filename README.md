# Classifier-for-Letter-Data-2
The purpose of this project is to build a classifier capable of recognising handwritten letters. 
The dataset is available at: http://ai.stanford.edu/~btaskar/ocr/.  We can get information about the data format.

Here, the data is accessible via a library data_extraction.py. The data has the same overall format as for digits:
- an X matrix, where each row represents a vectorized image.
- a vector Y whose coefficients represent the output, according to the following code: 0 = 'a', 1 = 'b',
... , 25 = 'z'
- an array numpy Z whose elements are matrices of 0 and 1 bitmap image of the letter
handwritten
In order to use these numpy arrays, we need to:
- copy the files letter.data and data_extraction.py into a directory. - import data_extraction into your program.
Example: The element with index 1234 is a 'b', which corresponds to an output of 1 according to the code described above.

 import matplotlib.pyplot as plt
import numpy as np
i=0
X = []
Y = []
Z = []
with open("letter.data", 'r') as infile:
for line in infile:
line = line.split()
V = line[6:]
V = [int(v) for v in V]
V = np.array(V) X.append(V) Y.append(ord(line[1]) - 97) V = V.reshape(16,8)
 Z.append(V)
X = np.array(X)
Y = np.array(Y)
Z = np.array(Z)

X[1234]


[3]: Y[1234]
[3]: 1
[4]: plt.imshow(Z[1234])
[4]: <matplotlib.image.AxesImage at 0x1148fa160>

We need to copy the file 'letter.data' to your working directory. We can then import data_extraction.py to access the X, Y and Z tables.

array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
       0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
       0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
       1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0,
       0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
       1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0])
       
Exercice 1 :

Let's look at the data.
- Are the classes balanced (i.e. all classes have the same size)?
- Can we explain why, by looking at the way the dataset was constructed?
data set? (see link at the beginning)
- What is the size of the class corresponding to 'n'?
- Assuming that a classifier always answers 'n', what would be its error rate?
We will take this value as our baseline. (this means that a classifier that makes more errors has not learned anything)

Exercice 2 :

Using the files that have been produced previously:
- Separate the data into a training set (90%) and a test set (10%).
- Construct a linear classifier (consider loss-log-likelihood) that we will learn on the training set.
learn on the training set.
- What error rate does this classifier have on the test set?

Exercice 3 :

Same questions as above, with a 1-layer hidden neural network. We will test the following configurations:
- hidden layer with 10, 20, and 30 neurons.
- activation function: ReLU
- For these configurations, what is the error rate on the test set?


