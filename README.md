# Optical-Character-Recognition-using-Googles-Tesseract

Input Image:



![Input](img/Img2.png)


Image is resized to size: 640 X 640.
Text detection was done by EAST text detector pre trained model.

During Text detection position of text with respect to text center, angle is determined, and detected portion is cropped for further image processing.

![Text Detected](img/img2_detect.png)

After Text detection, detected portion is cropped.

![alt text](img/img99.png)

Image is preprocessed by sharpen filter,converted to gray scale, dialated, and eroded to get better text prediction.

![alt text](img/img100.png)

Finally image rotation is performed to get image in correct position.

![alt text](img/img101.png)


Predicted output: '1397769'



 
