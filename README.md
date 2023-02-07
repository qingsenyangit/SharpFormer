
### Installation

This implementation based on [BasicSR](https://github.com/xinntao/BasicSR) which is a open source toolbox for image/video restoration tasks. 

```python
python 3.6.9
pytorch 1.5.1
cuda 10.1
```



```
cd SharpFormer
pip install -r requirements.txt
python setup.py develop --no_cuda_ext
```



* prepare data

  * ```mkdir ./datasets/GoPro ```
  
  * download the [train](https://drive.google.com/drive/folders/1AsgIP9_X0bg0olu2-1N6karm2x15cJWE) set in ./datasets/GoPro/train and [test](https://drive.google.com/drive/folders/1a2qKfXWpNuTGOm2-Jex8kfNSzYJLbqkf) set in ./datasets/GoPro/test (refer to [MPRNet](https://github.com/swz30/MPRNet)) 
  * it should be like:
  
    ```bash
    ./datasets/
    ./datasets/GoPro/
    ./datasets/GoPro/train/
    ./datasets/GoPro/train/input/
    ./datasets/GoPro/train/target/
    ./datasets/GoPro/test/
    ./datasets/GoPro/test/input/
    ./datasets/GoPro/test/target/
    ```
  
  * ```python scripts/data_preparation/gopro.py```
  
    * crop the train image pairs to 512x512 patches.


* eval

  * ```python basicsr/test.py -opt options/test/GoPro/SharpFormer-GoPro.yml  ```
  
* train

  * ```python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/GoPro/SharpFormer.yml --launcher pytorch```
