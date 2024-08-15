# feather-study-about-DIMO-MM
### Install
* use `cvtorchvision.cvtransforms` instead of torchvision.transforms, run `pip install opencv-torchvision-transforms-yuzhiyang` to install the package
### Use of the dataset
The original dataset uses a large number of features, I have provided both the original data and the pca data for quick verification.
* The pca dataset can be downloaded from: https://drive.google.com/file/d/1Ijss-X50rg2_RH0I5E9_rT8_UqmPKBO0/view?usp=drive_link
* The original dataset can be downloaded from:
* Unzip the downloaded data in the initial folder (in the same folder as quickstar.ipynb).
* This model uses the data mart lmdb format. If want to complete data set production. Please run [selftry1.py](https://github.com/schlange-git/feather-study-about-DIMO-MM/blob/main/datasets/BigEarthNet/selftry1.py) and Change code line278-280 to your own save path 
### Run the code

* We need to download the pre-trained weights of the original authors: https://huggingface.co/wangyi111/dino-mm/resolve/main/B14_vits8_dinomm_ep99.pth . Then put it in the checkpoints folder
