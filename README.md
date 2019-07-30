### Deep-learning the Latent Space of Light Transport
Created by <a href="https://www.uni-ulm.de/en/in/mi/institute/mi-mitarbeiter/pedro-hermosilla-casajus/" target="_blank">Pedro Hermosilla</a>, <a href="https://www.uni-ulm.de/in/mi/institut/mitarbeiter/sebastian-maisch/">Sebastian Maisch</a>, <a href="http://www.homepages.ucl.ac.uk/~ucactri/">Tobias Ritschel</a>, <a href="https://www.uni-ulm.de/in/mi/institut/mi-mitarbeiter/tr/" target="_blank">Timo Ropinski</a>.

![teaser](https://github.com/viscom-ulm/GINN/blob/master/teaser/teaser.jpg)

This repository contains the code of our <a href="https://arxiv.org/abs/1811.04756">EGSR paper</a>, **Deep-learning the Latent Space of Light Transport**. A video of our method can be found in the following <a href="https://www.youtube.com/embed/deLJvw10AaU">link</a>.

### Citation

If you find this code useful please consider citing us:

    @article{hermosilla2018ginn,
        title={Deep-learning the Latent Space of Light Transport},
        author={Pedro Hermosilla and Sebastian Maisch and Tobias Ritschel and Timo Ropinski },
        journal={Computer Graphics Forum (Proc. EGSR 2019)}
    }

### Pre-requisites

    numpy
    pygame
    CUDA 9.0
    tensorflow 1.12

### Installation

The first step is downloading the code for Monte Carlo Convolutions (MCCNN) in the following <a href="https://github.com/viscom-ulm/MCCNN">link</a>. The software expects the code to be in a folder named MCCNN. The second step is following the instruction on the README from MCCNN to compile the library.

### Real-time Viewer

Modify the compiling script in folder `rt_viewer/cuda_ops` with your cuda and python3 paths. Then, execute the compiling script to generate the CUDA/OpenGL operations. Lastly execute the scripts `rt_viewer/sss.sh` or `rt_viewer/gi.sh` to use the trained networks to visualize two 3D models.

### Training

In order to train a network on our datasets first download the data from the following link (COMMING SOON). Then, execute the script `processData.py` to generate the numpy files. Lastly, execute the command:

    python GITrainRT.py --useDropOut --useDropOutConv --augment --dataset 2

The parameter dataset determines which effect the network should be trained on.