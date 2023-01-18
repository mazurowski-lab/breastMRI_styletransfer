# StyleMapper: Deep Learning for Breast MRI Style Transfer with Limited Training Data

This is the code for our paper *Deep Learning for Breast MRI Style Transfer with Limited Training Data*, Journal of Digital Imaging 2022.

Code built on [https://github.com/NVlabs/MUNIT](https://github.com/NVlabs/MUNIT).

## Citation

If you use this code, please cite our paper:

```bib
@article{cao2022deep,
  title={Deep Learning for Breast MRI Style Transfer with Limited Training Data},
  author={Cao, Shixing and Konz, Nicholas and Duncan, James and Mazurowski, Maciej A},
  journal={Journal of Digital Imaging},
  pages={1--13},
  year={2022},
  publisher={Springer}
}
```

## File Descriptions

- `commands.txt`: example commands for training and experiments/testing
- `requirements.txt`: required packages that you (may) need to install
- `train.py`: script for training the model
- `test.py`: script for testing the model
- `style_code.py`: script for extracting style codes from images using trained model
- `compare_stylecodes.py`: script for comparing style codes of images, used for experiments in the supplementary material
- `validation.py`: script with validation utilities
- `trainer.py`: script with training utilities
- `networks.py`: script with model building utilities
- `data.py`: script for loading data
- `transforms.py`: script with image transformations used by model
- `utils.py`: script with miscellaneous utilities
- `configs`: Config files for training and testing with the same settings as in the paper. See `commands.txt` for examples of how to use these.

## Data Setup Instructions

All data is from the Duke Breast Cancer MRI dataset, linked [here](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=64685580). Once the data is downloaded, it can be sorted by MRI manufacturer (GE or Siemens) via the third column of [https://wiki.cancerimagingarchive.net/download/attachments/70226903/Clinical_and_Other_Features.xlsx](https://wiki.cancerimagingarchive.net/download/attachments/70226903/Clinical_and_Other_Features.xlsx).

Next, you can create folders named `datasets/breast_mri/GE` and `datasets/breast_mri/SIEMENS` in the base directory to house post-contrast MRI slice DICOM files for scans from these two scanner manufacturers, downloaded via the previous links. You can divide these files into data subsets with folders in `datasets/breast_mri/GE` of `trainA`, `trainB`, `testA`, `testB`, `validationA`, `validationB`, where `A` and `B` are used to split a subset in half (to more easily load into the model), e.g. the training set. For example, an MRI slice image in the training set could be `datasets/breast_mri/GE/trainA/Breast_MRI_438_post_1_100.dcm`.