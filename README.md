# Image Segmentation

This is the code repository of the following [paper](https://arxiv.org/pdf/2207.14626.pdf) to conduct a segmentation task under a biomecical circumstances.
"paper title"\
<em>Name</em>\
Journal Name, 2025.\
url

## Datasets

We perform experiments for image desnowing on Snow100K, combined image deraining and dehazing on Outdoor-Rain, and raindrop removal on the RainDrop datasets. To train multi-weather restoration, we used the AllWeather training set from TransWeather, which is composed of subsets of training images from these three benchmarks.

A public retinal vessel reference dataset CHASE_DB1 made available by Kingston University, London in collaboration with St. George’s, University of London. This is a subset of retinal images of multi-ethnic children from the Child Heart and Health Study in England (CHASE) dataset. This subset contains 28 retinal images captured from both eyes from 14 of the children recruited in the study. In this subset each retinal image is also accompanied by two ground truth images. This is provided in the form of two manual vessel segmentations made by two independent human observers for each of the images, in which each pixel is assigned a "1" label if it is part of a blood vessel and a "0" label otherwise. Making this subset publicly available allows for the scientific community to train and test computer vision algorithms (specifically vessel segmentation methodologies). Most importantly this subset allows for performance comparisons - several algorithms being evaluated on the same database allows for direct comparisons of their performances to be made.



We perform experiments on a public rentinal vessel reference dataset [CHASEDB1](https://researchdata.kingston.ac.uk/96/), which is made available by Kingston Univesity, London in collaboration with St. George's University of London. 

## Saved Model Weights

We share a pre-trained CHASEDB1 medical image segmentation model ...(link) with the network configuration in `models/archs/config.yml`. To evaluate ...(model name) using the pre-trained model checkpoint with the current version of the repository:
```bash
python val.py --name archs
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

* https://github.com/ermongroup/ddim
* https://github.com/bahjat-kawar/ddrm
* https://github.com/JingyunLiang/SwinIR
