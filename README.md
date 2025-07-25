# GDCUnet

This is the code repository of the following [paper](https://arxiv.org/abs/2507.18354) to conduct a segmentation task under a biomecical circumstances.
"Deformable Convolution Module with Globally Learned Relative Offsets for Fundus Vessel Segmentation"\
<em>Lexuan Zhu, Yuxuan Li, Yuning Ren</em>\
2025.\
https://arxiv.org/abs/2507.18354



## Abstract
Deformable convolution can adaptively change the shape of convolution kernel by learning offsets to deal with complex shape features. We propose a novel plug-and-play deformable convolutional module that uses attention and feedforward networks to learn offsets, so that the deformable patterns can capture long-distance global features. Compared with previously existing deformable convolutions, the proposed module learns the sub-pixel displacement field and adaptively warps the feature maps across all channels (rather than directly deforms the convolution kernel), which is equivalent to a relative deformation of the kernel’s sampling grids, achieving global feature deformation and the decoupling of (kernel size)–(learning network). Considering that the fundus blood vessels have globally self-similar complex edges, we design a deep learning model for fundus blood vessel segmentation, GDCUnet, based on the proposed convolutional module. Empirical evaluations under the same configuration and unified framework show that GDCUnet has achieved state-of-the-art performance on public datasets. Further ablation experiments demonstrated that the proposed deformable convolutional module could more significantly learn the complex features of fundus blood vessels, enhancing the model’s representation and generalization capabilities. The proposed module is similar to the interface of conventional convolution, we suggest applying it to more machine vision tasks with complex global self-similar features.



## Requirements

* pandas==1.5.3
* scikit-learn==1.2.2
* numpy==1.23.5
* matplotlib==3.7.0
* torch==2.0.0
* reformer-pytorch==1.4.4

## Prepare Dataset

We perform experiments on a public rentinal vessel reference dataset [CHASEDB1](https://researchdata.kingston.ac.uk/96/), the dataset directory should look like this:

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

## Experiments

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
    <td align="center"><b>GDCUnet Setting 3</td>
    <td align="center"><b>GDCUnet Setting 5</td>
    <td align="center"><b>Rolling UNet</td>
    <td align="center"><b>UNet</td>

  <tr>
    <td><img width='180' height="180" alt="06R" src="https://github.com/user-attachments/assets/60cf13f1-e313-4055-a385-db2a8257a275">
  </td>
    <td> <img width="180" height="180" alt="Ground Truth" src="https://github.com/user-attachments/assets/0a2cc6c4-cbcb-4731-ae06-a6ed3f3a985c"> 
  </td>
    <td> <img width="180" height="180" alt="GDCUnet Setting 3" src="https://github.com/user-attachments/assets/970b3e17-8456-4f84-b45d-7e871a7e8c55"> 
  </td>
    <td> <img width="180" height="180" alt="GDCUnet Setting 5" src="https://github.com/user-attachments/assets/fd6807c3-7ac2-4fe6-bd5a-ac1fb7ff39ce"> 
  </td>
    <td> <img width="180" height="180" alt="Rolling UNet" src="https://github.com/user-attachments/assets/62259e04-d675-428b-976d-a1174c73e30a">
    
  </td>
    <td> <img width="180" height="180" alt="UNet" src="https://github.com/user-attachments/assets/dfb457b2-4e5a-4891-a0c4-0e9cc810a6a9">
  </td>
    
  <tr>


  <tr>
    <td> <img width="180" height="180" alt="images" src="https://github.com/user-attachments/assets/81970617-c6d0-45e2-b299-244b7ddac692">
  </td>
    <td> <img width="180" height="180" alt="Ground Truth" src="https://github.com/user-attachments/assets/599b0699-d409-40aa-80a7-fc669baa9500">
  </td>
    <td> <img width="180" height="180" alt="GDCUnet Setting 3" src="https://github.com/user-attachments/assets/bd940568-ecf7-4c1e-996e-53979acbf178">
  </td>
    <td> <img width="180" height="180" alt="GDCUnet Setting 5" src="https://github.com/user-attachments/assets/ecde4be0-c42c-4dea-b0c5-992b6bb4bfe8">
  </td>
    <td> <img width="180" height="180" alt="Rolling UNet" src="https://github.com/user-attachments/assets/85603e57-3344-43ce-818a-b8960982949b">
    
  </td>
    <td> <img width="180" height="180" alt="UNet" src="https://github.com/user-attachments/assets/ca879939-25bc-4863-8933-f8bb75a70c78" >

  </td>
    
  <tr>
<table>

<table border='0' cellspacing='0' cellpadding='0'>
<tr>
    <td align="center"><b>UNet++</td>
    <td align="center"><b>Att-Unet</td>
    <td align="center"><b>Unext</td>
    <td align="center"><b>Uctransnet</td>
    <td align="center"><b>DconnNet</td>
    <td align="center"><b>DSCNet</td>
<tr>
    <td> <img width="180" height="180" alt="UNet++" src="https://github.com/user-attachments/assets/46f2599d-c0fb-4d0c-a78a-fc4574a92ce7"> 
 </td>
    <td> <img width="180" height="180" alt="Att-Unet" src="https://github.com/user-attachments/assets/7a16815f-9027-4ebf-9f95-af09fc3cd9fd"> 
 </td>
     <td> <img width="180" height="180" alt="Unext" src="https://github.com/user-attachments/assets/f882167a-e639-438f-a673-2a944410b420"> 
</td> 
    <td><img width="180" height="180" alt="Uctransnet" src="https://github.com/user-attachments/assets/13cb2833-ec08-45f1-b7a2-ce634939edd1">
</td>
    <td><img width="180" height="180" alt="DcnnNet" src="https://github.com/user-attachments/assets/53fa3b13-a29b-4d67-88c2-14e104600579">
</td>
    <td><img width="180" height="180" alt="DSCNet" src="https://github.com/user-attachments/assets/8e4f6171-b399-41da-a6a5-47f4e9b31337">
</td>
    

<tr>
    <td> <img width="180" height="180" alt="UNet++" src="https://github.com/user-attachments/assets/115eaaa7-b4c2-4683-9e9c-15e41e62f27d" >
 </td>
    <td> <img width="180" height="180" alt="Att-Unet" src="https://github.com/user-attachments/assets/6a2040a0-2bcc-4196-b948-b86852b4b50f"> 
 </td>
     <td> <img width="180" height="180" alt="Unext" src="https://github.com/user-attachments/assets/5616851d-82dd-43be-9e4a-49f05fa44464"> 
</td> 
    <td><img width="180" height="180" alt="Uctransnet" src="https://github.com/user-attachments/assets/6efab025-ed4c-47f1-9f66-fafd20453ef9">
</td>
    <td><img width="180" height="180" alt="DcnnNet" src="https://github.com/user-attachments/assets/99834bf6-c4c8-4dd7-a8e7-8e8a828a638d">
</td>
    <td><img width="180" height="180" alt="DSCNet" src="https://github.com/user-attachments/assets/c2d68c01-5c0a-4cb8-8dc4-1bd232fe9fec">
</td>
<tr>
    
<tr>
<table>


## Reference
If you use this code or models in your research and find it helpful, please cite the following paper:
```
@misc{zhu2025deformableconvolutionmoduleglobally,
      title={Deformable Convolution Module with Globally Learned Relative Offsets for Fundus Vessel Segmentation}, 
      author={Lexuan Zhu and Yuxuan Li and Yuning Ren},
      year={2025},
      eprint={2507.18354},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.18354}, 
}
```

## Acknowledgments

This work was supported by the Ministry of Education of the People’s Republic of
China under Grant 2024GH-ZDA-GJ-Y-09.

Parts of this code repository is based on the following works:

* [https://github.com/IGITUGraz/WeatherDiffusion](https://github.com/jeya-maria-jose/UNeXt-pytorch)
* [https://github.com/ZhangJC-2k/DPU](https://github.com/Jiaoyang45/Rolling-Unet)


