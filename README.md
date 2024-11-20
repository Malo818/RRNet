# CMRR: 2D-3D Cross Modal Regional Retrieval

This is the official PyTorch implementation of the following publication:

## 3D color scene datasets
**Download**  
SceneNN: [https://hkust-vgd.github.io/scenenn/](https://hkust-vgd.github.io/scenenn/)  
S3DIS: [https://github.com/alexsax/2D-3D-Semantics](https://github.com/alexsax/2D-3D-Semantics)  
ScanNet: [http://www.scan-net.org/](http://www.scan-net.org/)

The SceneNN, S3DIS and ScanNet datasets were collected and collated to yield 95, 272 and 707  
colored point clouds respectively.  Each point cloud is segmented into 32 sub-regions, which  
are subsequently projected into a four-view image.  In addition, voxel subsampling was performed  
on each scene, followed by equal-density folding according to a density of 0.06.  The model is  
trained with multiple views and multiple point clouds.

## Repo Structure
*   `data`: Generate and process datasets
    * `build_dataset`:multi-view image and fold point cloud processing

*   `logs`: Save training information

*   `models`: Networks and layers
     * `imagenet`: image feature extraction
     * `pointnet`: point cloud coding and decoding
     * `transformer`: location coding and multimodal feature fusion
     * `feature_transformation`: multi-view information fusion
     * `network.`: The backbone of RRNet

*   `utils`: data processing and mapping

*   `config`: the main configuration information of the model

*   `train`:  


    $ python train.py --config config.json --logdir logs/RR

Log files and network parameters will be saved to the `logs/RR` folder.
*   `test`:  


    $ python test.py

