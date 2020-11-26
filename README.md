# NCTU_hw2

**Requirements:**

Python 3.6  
PyTorch 1.5

**Setup:**  
1. Download SVHN Dataset  including train folder and test folder
2. Create a folder naming models in the current folder


**Usage:**  
1. run mat_to_csv.py to transform .mat file to .csv
2. run train.py to train the model , the model is from torch : faster_rcnn , and save the model to ./models
3. run infer.py and save the prediction results to the pred_results.json in the current folder



**References:**  
https://github.com/penny4860/SVHN-deep-digit-detector  
https://github.com/danielzgsilva/SVHN_Project  
https://github.com/israelwei/SVHN_Classic  
https://github.com/veax-void/digit_object_detection


