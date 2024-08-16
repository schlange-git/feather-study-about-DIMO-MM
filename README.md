# Feather-study-about-DIMO-MM
## Install
* **Special attention:** use `cvtorchvision.cvtransforms` instead of torchvision.transforms, run `pip install opencv-torchvision-transforms-yuzhiyang` to install the package
* For additional environment information, see environment.yml
## Use of the dataset
### Used resource sources
* I'm using the BigEarthNet dataset, the original data site is: https://bigearth.net/
* Link to the original modelhttps: [//github.com/zhu-xlab/DINO-MM](https://github.com/zhu-xlab/DINO-MM)
### Configuration of the dataset and pre-trained weights
The original dataset uses a large number of features, I have provided both the original data and the pca data for quick verification.
* The pca dataset can be downloaded from: https://drive.google.com/file/d/1Ijss-X50rg2_RH0I5E9_rT8_UqmPKBO0/view?usp=drive_link
* The original dataset can be downloaded from: https://drive.google.com/file/d/12Q4ddan2cUdyS7dx2BcgnWvQJdSnjzlB/view?usp=drive_link
* Unzip the downloaded data in the initial folder (in the same folder as quickstar.ipynb).
* We need to download the pre-trained weights of the original authors: https://huggingface.co/wangyi111/dino-mm/resolve/main/B14_vits8_dinomm_ep99.pth . Then put it in the checkpoints folde
### make lmdb dataset 
The lmdb dataset is the basis for running visualisation as well as resnet-18. Approximately 100GB of space is pre-allocated for the lmdb in the code.
* This model uses the data mart lmdb format. Please run [selftry1.py](https://github.com/schlange-git/feather-study-about-DIMO-MM/blob/main/datasets/BigEarthNet/selftry1.py) and Change code line278-280 to your own save path. Change lines 147 and 148 to point to the BigEarthNet dataset.
## Run the code
### attention
1. **%%%%%%task.py** and **quickstart.ipynb** are similar in content. But the order of the blocks is not the same. The former is the order I wrote myself. The latter is more conducive to fast verification. I put the data preparation and other pre-work code block to the end.
2. In addition, the parameters are obtained in the code to quickly run the code. So the final numbers are not the values in the paper. For results similar to those in the paper, see **parameter.md**. With the help of an RTX4090 workstation with 48GB memory and 64 core CPU, the whole process takes about 30 hours.
### process

* Using only the dataset after pca is the default configuration for ipynb. If the original data (about 14GB) is used, change the name in read file code in the third block.
* The code in quickstart.ipynb should run directly. Please let me know if it doesn't work.
