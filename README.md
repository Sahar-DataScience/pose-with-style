# Garment transfer and pose transfer using infrence mode of [pose with style](https://pose-with-style.github.io)

<p align='center'>
<img src='https://github.com/Sahar-DataScience/pose-with-style/blob/main/data/resized_img.png' width='30%'/>
<img src='https://github.com/Sahar-DataScience/pose-with-style/blob/main/data/fashionWOMENSkirtsid0000177102_1front.png' width='30%'/>
<img src='https://github.com/Sahar-DataScience/pose-with-style/blob/main/data/output/fashionWOMENSkirtsid0000177102_1front_and_resized_img_upper_body_vis.png' width='30%'/>
</p>


## Requirements
```
conda create -n posewithstyle python=3.6
conda activate posewithstyle
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```
Intall openCV using `conda install -c conda-forge opencv` or `pip install opencv-python`.
If you would like to use [wandb](https://wandb.ai/site), install it using `pip install wandb`.

## Download pretrained models
You can download the pretrained model [here](https://filebox.ece.vt.edu/~Badour/pose-with-style/pretrained/posewithstyle.pt), and the pretrained coordinate completion model [here](https://filebox.ece.vt.edu/~Badour/pose-with-style/downloads/CCM_epoch50.pt).

Note: we also provide the pretrained model trained on [StylePoseGAN](https://people.mpi-inf.mpg.de/~ksarkar/styleposegan/) [Sarkar et al. 2021] DeepFashion train/test split [here](https://filebox.ece.vt.edu/~Badour/pose-with-style/pretrained/posewithstyle_sarkarsplit.pt). We also provide this split's pretrained coordinate completion model [here](https://filebox.ece.vt.edu/~Badour/pose-with-style/downloads/sarkar_CCM_epoch50.pt).

## Reposing
Download the [UV space - 2D look up map](https://filebox.ece.vt.edu/~Badour/pose-with-style/downloads/dp_uv_lookup_256.npy) and save it in `util` folder.

We provide sample data in `data` directory. The output will be saved in `data/output` directory.
```
python inference.py --input_path ./data --CCM_pretrained_model path/to/CCM_epoch50.pt --pretrained_model path/to/posewithstyle.pt
```

To repose your own images you need to put the input image (input_name+'.png'), dense pose (input_name+'_iuv.png'), and silhouette (input_name+'_sil.png'), as well as the target dense pose (target_name+'_iuv.png') in `data` directory.
```
python inference.py --input_path ./data --input_name fashionWOMENDressesid0000262902_3back --target_name fashionWOMENDressesid0000262902_1front --CCM_pretrained_model path/to/CCM_epoch50.pt --pretrained_model path/to/posewithstyle.pt
```

## Garment transfer
Download the [UV space - 2D look up map](https://filebox.ece.vt.edu/~Badour/pose-with-style/downloads/dp_uv_lookup_256.npy) and the [UV space body part segmentation](https://filebox.ece.vt.edu/~Badour/pose-with-style/downloads/uv_space_parts.npy). Save both in `util` folder.
The UV space body part segmentation will provide a generic segmentation of the human body. Alternatively, you can specify your own mask of the region you want to transfer.

We provide sample data in `data` directory. The output will be saved in `data/output` directory.
```
python garment_transfer.py --input_path ./data --CCM_pretrained_model path/to/CCM_epoch50.pt --pretrained_model path/to/posewithstyle.pt --part upper_body
```

To use your own images you need to put the input image (input_name+'.png'), dense pose (input_name+'_iuv.png'), and silhouette (input_name+'_sil.png'), as well as the garment source target image (target_name+'.png'), dense pose (target_name+'_iuv.png'), and silhouette (target_name+'_sil.png') in `data` directory. You can specify the part to be transferred using `--part` as `upper_body`, `lower_body`, or `face`. The output as well as the part transferred (shown in red) will be saved in `data/output` directory.
```
python garment_transfer.py --input_path ./data --input_name fashionWOMENSkirtsid0000177102_1front --target_name fashionWOMENBlouses_Shirtsid0000635004_1front --CCM_pretrained_model path/to/CCM_epoch50.pt --pretrained_model path/to/posewithstyle.pt --part upper_body
```


## Testing
To test the reposing model and generate the reposing results:
```
python test.py /path/to/DATASET --pretrained_model /path/to/step2/pretrained/model --size 512 --save_path /path/to/save/output
```
Output images will be saved in `--save_path`.

You can find our reposing output images [here](https://pose-with-style.github.io/results.html).

## Evaluation
We follow the same evaluation code as [Global-Flow-Local-Attention](https://github.com/RenYurui/Global-Flow-Local-Attention/blob/master/PERSON_IMAGE_GENERATION.md#evaluation).


## Bibtex
Please consider citing our work if you find it useful for your research:

	@article{albahar2021pose,
	    title   = {Pose with {S}tyle: {D}etail-Preserving Pose-Guided Image Synthesis with Conditional StyleGAN},
      author  = {AlBahar, Badour and Lu, Jingwan and Yang, Jimei and Shu, Zhixin and Shechtman, Eli and Huang, Jia-Bin},
	    journal = {ACM Transactions on Graphics},
      year    = {2021}
	}


## Acknowledgments
This code is heavily borrowed from [Rosinality: StyleGAN 2 in PyTorch](https://github.com/rosinality/stylegan2-pytorch).
