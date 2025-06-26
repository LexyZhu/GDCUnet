# Image Segmentation

This is the code repository of the following [paper](https://arxiv.org/pdf/2207.14626.pdf) to conduct a segmentation task under a biomecical circumstances.
"paper title"\
<em>Name</em>\
Journal Name, 2025.\
url



## Datasets

We perform experiments for medical image segmentation on ...(dataset link) dataset. 

## Saved Model Weights

We share a pre-trained CHASEDB1 medical image segmentation model ...(link) with the network configuration in `models/archs/config.yml`. To evaluate ...(model name) using the pre-trained model checkpoint with the current version of the repository:
```bash
python val.py --name archs
```

Check out below for some visualizations of our model outputs.

## Image Desnowing

<table border='0' cellspacing='0' cellpadding='0'>
  <tr>
    <td align="center"><b>Ground Truth</td>
    <td align="center"><b>Archs</td>
    <td align="center"><b>Rolling UNet</td>
    <td align="center"><b>UNet</td>
    <td align="center"><b>UNet++</td>
    
  <tr>
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
