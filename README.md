# LGNet: Explicit Local-Global Feature Modeling for Cloud Removal

Ye Deng,  Zhihao Li, Peng Wang, Weiping Ding, Long Chen, Hangjun Ye, Weilong Wang, Xiaoshuai Hao*, 2026

<!--   
  
\- \*\*2026-03-25:\** Readme and config are released. 🎊🎊🎊   
\- pending 1:\** Network. 🎊🎊🎊   
\- pending 2:\** pretrain-models. 🎊🎊🎊   
  
\---  
  
\> \*\*Abstract:\** Designing neural network architectures that can effectively extract feature information from remote sensing images is crucial for achieving high-quality cloud removal. The feature information in remote sensing images can generally be categorized into local and global features. Existing methods typically handle local features well using window attention or convolution operations, but they face challenges when dealing with global features. In this paper, we first propose Dual Gated Convolution (DGC), which efficietly learns the non-local dependencies of images in the frequency domain by leveraging Fourier theory. Based on DGC, we design a Local-Global Block (LGBlock), which explicitly models both local and global dependencies by combining window attention with adaptive convolutional mechanisms. Furthermore, we develop LGNet (Local-Global Network), a U-Net-style cloud removal framework that leverages the strengths of LGBlock to achieve efficient featu  
\> re modeling. Experimental results on several public datasets demonstrate that LGNet out  
\> performs existing methods.  
  
\!\[\](figs/LGNet.png)  
  
  
\---  
  
  
  
\## ⚙️ Installation  
  
\- Python 3.9.0  
  
\- PyTorch 1.13.1  
  
\- NVIDIA GPU + \[CUDA\](https://developer.nvidia.com/cuda-downloads) 11.7  
  
\- Clone this repo:  
  
  \`\`\`bash  
  git clone https://github.com/LGNet.git  
  cd LGNet  
  \`\`\`  
  
\- Create conda environment:  
  
  \`\`\`bash  
  conda env create -f LGNet-env.yml -n LGNet  
  conda activate LGNet   
  \`\`\`  
  
  
  
\## 🔗 Contents  
  
1\. \[Datasets\](#datasets)  
1\. \[Models\](#models)  
1\. \[Training\](#training)  
1\. \[Testing\](#testing)  
1\. \[Results\](#results)  
1\. \[Citation\](#citation)  
1\. \[Acknowledgements\](#acknowledgements)  
  
\---  
  
  
  
\## <a name="datasets"></a>🖨️ Datasets  
\- \`\`\`RICE-I\`\`\`: It consists of 500 pairs of filmy and cloud-free images obtained from Google Earth.   
\[RICE-I\](https://github.com/BUPTLdy/RICE_DATASET)  
\- \`\`\`RICE-II\`\`\`: It consists of 736 pairs of images captured by Landsat 8 OLI/TIRS, including cloudy, cloudless, and mask images. The mask images were created using the Landsat Level-1 quality band to identify regions affected by clouds, cloud shadows, and cirrus clouds. The cloudless images were captured at the same location as the corresponding cloud images with a maximum interval of 15 days. \[RICE-II\](https://github.com/BUPTLdy/RICE_DATASET)  
\- \`\`\`T-CLOUD\`\`\`: The T-CLOUD, a real scene thin cloud dataset captured from Landsat 8 RGB images, contains 2,939 image pairs. The cloudy images and their clear counterparts are separated by one satellite re-entry period (16 days). These images are carefully selected with similar lighting conditions and are cropped into $256 \times 256$ patches. The dataset is split into 2351 images for training and 588 for testing.   
\[T-CLOUD\](https://github.com/haidong-Ding/Cloud-Removal)  
\- \`SEN12MS-CR\`:It contains approximately 110,000 samples from 169 distinct, non-overlapping regions across various continents and meteorological seasons. Each sample includes a pair of Sentinel-2 images, one cloudy and one cloud-free, along with the corresponding Sentinel-1 synthetic aperture radar (SAR) image. \[SEN12MS-CR\](https://mediatum.ub.tum.de/1554803)  
  
  
Download training and testing datasets and put them into the corresponding folders of \`./datasets/\`.  
  
  
  
\## <a name="models"></a>📦 Models  
  
\[Rice1 pretrained model Baidu Disk\](https://pan.baidu.com/s/1oK_UyIMFdLFq7MzsaRBa_A?pwd=yg1a)  
  
\[Rice2 pretrained  model Baidu Disk\](https://pan.baidu.com/s/1rLhQ41Qli23L_D0H8nI2vA?pwd=ngke)  
  
\[T-CLOUD pretrained model Baidu Disk\](https://pan.baidu.com/s/10VyNIwJDyGNSoqE0UlDAvA?pwd=ebmg)  
  
  
\## <a name="training"></a>🔧 Training  
\- The training configuration is in folder \`option/RICE1-CR-Former.yml\`,  \`option/RICE2-CR-Former.yml\`, and \`option/rice1-LGNet.yml\`.  
  
    \- The argument \`dataroot_gt\` in the \`.yml\` file specifies the path to the training dataset.  
    \- The argument \`name\` in the \`.yml\` file specifies the path to the training model (*.pth). By default, it is set to the \`./experiments/name\` folder.  
    \- The training experiments are stored in the \`./experiments\` directory.  
  
\- Run the following scripts for training  
  \`\`\`shell  
  \# RICE1, input=256*256, 1 A100 GPU  
  sh train_script/rice1.sh  
    
  \# RICE2, input=256*256, 1 A100 GPU  
  sh train_script/rice2.sh  
  
  \# T-CLOUD, input=256*256, 1 A100 GPU  
  sh train_script/t_cloud.sh  
  \`\`\`  
  \`\`\`  
   \- Set \`-opt\` as the model configuration option in the '.sh' command.  
  \`\`\`  
  
\## <a name="testing"></a>🔨 Testing  
\- Download the pre-trained \[models\](https://pan.baidu.com/s/1TVm62brZw6Rd85bhgPhwUA?pwd=kewc) and place them in \`./experiments\` directory.  
  
\- Use RICE1, RICE2, and T-CLOUD testing datasets.  
  
\- Test the model.   
Run the following scripts for training  
  \`\`\`shell  
  \# RICE1  
  sh test_script/rice1_test.sh  
  
  \# RICE2  
  sh test_script/rice2_test.sh  
  
  \# T-CLOUD  
  sh test_script/t_cloud_test.sh  
  \`\`\`  
  \`\`\`  
  \- Set \`--opt\` as the model configuration option.  
  \- Set \`--input_dir\` as the input test dataset path.  
  \- Set \`--input_truth_dir\` as the input test dataset ground truth path.  
  \- Set \`--result_dir\` as output image path.  
  \- Set \`--weights\` as the pretrained-model path.  
  \- The default results will be saved under the \*output* folder.  
  \`\`\`  
  
\## <a name="results"></a>🔎 Results  
  
We achieved state-of-the-art performance. Detailed results can be found in the paper.  
  
<details>  
Quantitative Comparison (click to expand)  
  
\- Quantitative comparison results on the four datasets in Table I of the main paper. Metrics marked with ↓ are better when lower, and those with ↑ are better when higher. Best results are shown in bold.  
  
<p align="center">  
  <img width="900" src="figs/table1.png">  
</p>  
  
</details>  
  
<details>  
Visual Comparison (click to expand)  
  
\- RICE1 and RICE2 results in Figure 4 of the main paper  
  
<p align="center">  
  <img width="900" src="figs/Figure4.png">  
</p>  
  
  
  
  
</details>  
  
  
  
\## <a name="citation"></a>📎 Citation  
  
If you find the code helpful in your research or work, please cite the following paper(s).  
  
\`\`\`  
@ARTICLE{wu2024cr,  
	author={Ye Deng,  Zhihao Li, Peng Wang, Weiping Ding, Long Chen, Hangjun Ye, Weilong Wang, Xiaoshuai Hao},  
	title={LGNet: Explicit Local-Global Feature Modeling for Cloud Removal}, 

        journal={Applied Soft Computing},

       pages={115401},

       year={2026},

       publisher={Elsevier}  
	pages={1-14}}  
\`\`\`  
  
\## License  
  
  
The codes and the pre-trained models in this repository are under the MIT license as specified by the LICENSE file.  
This code is for educational and academic research purpose only.  
  
\## <a name="acknowledgements"></a>💡 Acknowledgements  
  
This code is built on \[Restormer\](https://github.com/swz30/Restormer). We thank the authors for sharing their codes.  
