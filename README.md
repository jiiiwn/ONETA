# Oneta: Multi-Style Image Enhancement Using Eigentransformation Functions
<em>Jiwon Kim<sup>1*</sup></em>, <em>Soohyun Hwang<sup>1*</sup></em>, *Dong-O Kim<sup>2</sup>, Changsu Han<sup>2</sup>, Min Kyu Park<sup>2</sup>, Chang-Su Kim<sup>1</sup>*
![overview image](overview_image.png)



# Abstract
The first algorithm, called Oneta, for a novel task of multi-style image enhancement is proposed in this work. 
Oneta uses two point operators sequentially: intensity enhancement with a transformation function (TF) and color correction with a color correction matrix (CCM). 
This two-step enhancement model, though simple, achieves a high performance upper bound. Also, we introduce eigentransformation function (eigenTF) to represent TF compactly. 
The Oneta network comprises Y-Net and C-Net to predict eigenTF and CCM parameters, respectively. To support $K$ styles, Oneta employs $K$ learnable tokens. 
During training, each style token is learned using image pairs from the corresponding dataset. In testing, Oneta selects one of the $K$ style tokens to enhance an image accordingly. 
Extensive experiments show that the single Oneta network can effectively undertake six enhancement tasks --- retouching, image signal processing, low-light image enhancement, dehazing, underwater image enhancement, and white balancing --- across 30 datasets.

# Checkpoint Download
(https://drive.google.com/file/d/1PlWTUALfFaMXCjaSWI1kYSPQAoFABIwV/view?usp=drive_link)

# Environment Setting
```bash
pip install -r requirements.txt 
```

# Path Setting in Code
Please update the paths listed below in the code to match your local environment.
1. In main.py, u_path, rgb_test_dir, gt_test_dir, save_dir
2. In inference.py, ckpt_path
3. In util.py, meta_data_dir

# Style token & File extension for each Dataset
We assigned a unique integer representing the style to each dataset. 
Refer to the [table](table.png) for the file extension used by each dataset. 
If you change any of the extensions, make sure to update the prefix_extract function in util.py accordingly.
