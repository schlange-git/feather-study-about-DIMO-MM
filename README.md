# feather-study-about-DIMO-MM
### Install
* use `cvtorchvision.cvtransforms` instead of torchvision.transforms, run `pip install opencv-torchvision-transforms-yuzhiyang` to install the package
### Use of the dataset
* This model uses the data mart lmdb format. If want to complete data set production. Please run file
  ```
  datasets/BigEarthNet/selftry1.py
  ```
  and Change code line278-280 to your own save path

### Run the code
* Because the original dataset uses a large number of features, I have provided both the original data and the pca data for quick verification.
* 
