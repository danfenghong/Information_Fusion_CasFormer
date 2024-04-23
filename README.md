<div align="center">
<h1>CasFormer: Cascaded transformers for fusion-aware computational hyperspectral imaging</h1>
  
Chenyu Li, [Bing Zhang](https://scholar.google.com/citations?user=nHup8tQAAAAJ&hl=en), [Danfeng Hong](https://scholar.google.com/citations?hl=en&user=n7gL0_IAAAAJ&view_op=list_works&sortby=pubdate), [Gemine Vivone](https://scholar.google.com/citations?user=sjb_uAMAAAAJ&hl=en), [Shutao Li](https://scholar.google.com/citations?user=PlBq8n8AAAAJ&hl=en), [Jocelyn Chanussot](https://scholar.google.com/citations?user=6owK2OQAAAAJ&hl=en)

**Information Fusion: (https://doi.org/10.1016/j.inffus.2024.102408).  
</div>

![alt text](./CasFormer.png)

**Fig.1.** The overall architecture of CasFormer with the input of RGB, coded measurement, and mask. The core module of **CasFormer** consists of a series of cascade-attention (CA) blocks, where “mask” is directly correlated with imaging devices.

## Code Running
Please simply run `./train_code/train.py` demo to reproduce our CAVE results by [CAVE dataset](http://www.cs.columbia.edu/CAVE/databases/multispectral) (Using [PyTorch](https://pytorch.org/) with `Python 3.7` implemented on `Windows` OS).

- Before: For the required packages, please refer to detailed `.py` files.
- Parameters: The trade-off parameters as `train_opt.lambda_*` could be better tuned and the network hyperparameters are flexible.
- Results: Please see the five evaluation metrics (PSNR, SSIM, and SAM) logged in `./checkpoints/CAVE_*name*/precision.txt` and the output `.mat` files saved in `./Results/CAVE/`.
- The experiments were run on 8 NVIDIA GeForce RTX 3090 GPUs.

:exclamation: The pretrained model on CAVE dataset can be downloaded from [here]() with code: y2a0 or [here]().

:exclamation: You may need to manually download the KAIST dataset ([Google drive]() and [Baiduyun]()  with code: 6q6j) and ICVL dataset ([Google drive]() and [Baiduyun]() with code:  6q6j) on your local in the folder under path `./dataset`, due to storage restriction, from the following links of google drive or baiduyun:

## Citation Details

**Please kindly cite the papers if this code is useful and helpful for your research.**

```
@article{LI2024102408,
 title = {CasFormer: Cascaded transformers for fusion-aware computational hyperspectral imaging},
 journal = {Information Fusion},
 volume = {108},
 pages = {102408},
 year = {2024}
}
```

Licensing
---------

Copyright (C) 2024 Danfeng Hong

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.

Contact Information:
--------------------

Danfeng Hong: hongdanfeng1989@gmail.com<br>
Danfeng Hong is with the Aerospace Information Research Institute, Chinese Academy of Sciences, 100094 Beijing, China.
