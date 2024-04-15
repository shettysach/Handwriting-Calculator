# Handwriting-Calculator

A calculator that uses handwritten digits and operators to calculate the result, using contour detection and CNN model prediction.
- Tensorflow (Keras) is used to create, train and load the neural network model used for predictions.
- CustomTKinter is used to provide the GUI.
- OpenCV and Pillow (PIL) are used to read input from the GUI canvas and to obtain contours for individual digits/operators.
- The individual digits/operators are detected and predicted. The predictions are combined into a string and evaluated to get the result.

![demo0](https://github.com/ShettySach/Handwriting-Calculator/assets/132273464/c8643ca4-2bad-46f7-ba14-13b01e7a9d2b)

#### Contour boxes (green), predicted values (blue) and accuracies (red)

![Contours](https://github.com/ShettySach/Handwriting-Calculator/assets/132273464/5e3e7121-21ca-4337-a86c-c5656c305bb2)

## Requirements -
```bash
pip install -r requirements.txt
```
* Tensorflow (Keras)
* OpenCV
* Pillow (PIL)
* Pandas
* Numpy
* CustomTkinter

## Instructions -
* Clone the repo and run the Jupyter notebook, **MAIN.ipynb** or run **MAIN.py**
* You can use digits `0 to 9`, operators `+ - × /`, decimal point `.` and parentheses `( )`
* You can also use ×× for exponentiation and // for floor division
  
### Data
* [MNIST dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)
* [Symbol dateset by Irfan Chahyadi ](https://github.com/irfanchahyadi/Handwriting-Calculator/blob/master/src/dataset/data.pickle)
