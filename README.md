# GDCUnet

This is the code repository of the following [paper](https://arxiv.org/pdf/2207.14626.pdf) to conduct a segmentation task under a biomecical circumstances.
"paper title"\
<em>Name</em>\
Journal Name, 2025.\
url

## Abstract
Deformable convolution can adaptively change the shape of convolution kernel by learning offsets to deal with complex shape features. We propose a novel plug-and-play deformable convolutional module that uses attention and feedforward networks to learn offsets, so that the deformable patterns can capture long-distance global features. Compared with previously existing deformable convolutions, the proposed module learns the sub-pixel displacement field and adaptively warps the feature maps across all channels (rather than directly deforms the convolution kernel), which is equivalent to a relative deformation of the kernel’s sampling grids, achieving global feature deformation and the decoupling of (kernel size)–(learning network). Considering that the fundus blood vessels have globally self-similar complex edges, we design a deep learning model for fundus blood vessel segmentation, GDCUnet, based on the proposed convolutional module. Empirical evaluations under the same configuration and unified framework show that GDCUnet has achieved state-of-the-art performance on public datasets. Further ablation experiments demonstrated that the proposed deformable convolutional module could more significantly learn the complex features of fundus blood vessels, enhancing the model’s representation and generalization capabilities. The proposed module is similar to the interface of conventional convolution, we suggest applying it to more machine vision tasks with complex global self-similar features.



## Create Environment

* pandas==1.5.3
* scikit-learn==1.2.2
* numpy==1.23.5
* matplotlib==3.7.0
* torch==2.0.0
* reformer-pytorch==1.4.4

## Prepare Dataset

We perform experiments on a public rentinal vessel reference dataset [CHASEDB1](https://researchdata.kingston.ac.uk/96/), which is made available by Kingston Univesity, London in collaboration with St. George's University of London. The downloaded dataset is shown in the following form:


    |--inputs
        |--images
            |--Image_01L.jpg
            |--Image_01R.jpg
            |--...
        |--masks
            |--0
                |--Image_01L_1stHO.png
                |--Image_01R_1stHO.png
                |--...

## Simulation Experiment

```bash
# archs
python train.py  --name GDCUnet --arch GDCUnet --epochs 4000

# Rolling Unet
python train.py --name Rolling_Unet --arch Rolling_Unet_S

# Attention Unet
python train.py --nae Attention_Unet --arch Attention_Unet

```




## Saved Model Weights

We share a pre-trained CHASEDB1 medical image segmentation model ...(link) with the network configuration in the following form:

    |--models
        |--GDCUnet
            |--config.yml
        |--Rolling_Unet
            |--config.yml
        |--...

To evaluate GDCUnet using the pre-trained model weights with the current version of the repository:

```bash
python val.py --name GDCUnet
```



Check out below for some visualizations of our model outputs.

## CHASEDB1 Medical Image Segmentation

<table border='0' cellspacing='0' cellpadding='0'>
  <tr>
    <td align="center"><b>Images</td>
    <td align="center"><b>Ground Truth</td>
    <td align="center"><b>Archs</td>
    <td align="center"><b>Rolling UNet</td>
    <td align="center"><b>UNet</td>
    <td align="center"><b>UNet++</td>
    
  <tr>
    <td><img width='180', alt="06R" src="https://github.com/user-attachments/assets/60cf13f1-e313-4055-a385-db2a8257a275">
  </td>
    <td> <img width="180" alt="Ground Truth" src="https://github.com/user-attachments/assets/0a2cc6c4-cbcb-4731-ae06-a6ed3f3a985c"> 
  </td>
    <td> <img width="180" alt="Archs" src="https://github.com/user-attachments/assets/18672231-896c-4f05-a6a4-7b6cf84f9c97"> 
  </td>
    <td> <img width="180" alt="Rolling UNet" src="https://github.com/user-attachments/assets/27e0897d-8517-41c7-99fb-b398c4eb420e">
    
  </td>
    <td> <img width="180" alt="UNet" src="https://github.com/user-attachments/assets/dfb457b2-4e5a-4891-a0c4-0e9cc810a6a9">
  </td>
    <td> <img width="180" alt="UNet++" src="https://github.com/user-attachments/assets/46f2599d-c0fb-4d0c-a78a-fc4574a92ce7"> 
 </td>
  <tr>
  <tr>
    <td> <img width="180" alt="08R" src="https://github.com/user-attachments/assets/83996a56-5962-4feb-a61b-dd486fcd5064">
  </td>
    <td> <img width="180" alt="Ground Truth" src="https://github.com/user-attachments/assets/018e6393-f824-4159-9fb1-9fd88c738cbb">
  </td>
    <td> <img width="180" alt="Archs" src="https://github.com/user-attachments/assets/bdd3489c-1ec4-40af-b0ce-5dbb605d4ba7">
  </td>
    <td> <img width="180" alt="Rolling UNet" src="https://github.com/user-attachments/assets/817d6767-82a4-44ef-aafa-a1980c6029d6">
    
  </td>
    <td> <img width="180" alt="UNet" src="https://github.com/user-attachments/assets/438c129f-c169-42b7-aac4-789d5ea5667a" >

  </td>
    <td> <img width="180" alt="UNet++" src="https://github.com/user-attachments/assets/ef8ac8c1-f9b1-4213-ab4e-8f2422769f9c" >
 </td>
  <tr>
<table>


## Reference
If you use this code or models in your research and find it helpful, please cite the following paper:
```
@article{ozdenizci2023,
  title={Restoring vision in adverse weather conditions with patch-based denoising diffusion models},
  author={Ozan \"{O}zdenizci and Robert Legenstein},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  pages={1-12},
  year={2023},
  doi={10.1109/TPAMI.2023.3238179}
}
```

## Acknowledgments

Authors of this work are affiliated with Graz University of Technology, Institute of Theoretical Computer Science, and Silicon Austria Labs, TU Graz - SAL Dependable Embedded Systems Lab, Graz, Austria. This work has been supported by the "University SAL Labs" initiative of Silicon Austria Labs (SAL) and its Austrian partner universities for applied fundamental research for electronic based systems.

Parts of this code repository is based on the following works:

* https://github.com/IGITUGraz/WeatherDiffusion
* https://github.com/ZhangJC-2k/DPU
* 

