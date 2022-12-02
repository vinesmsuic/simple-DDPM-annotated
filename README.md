# Simple DDPM

A simplified DDPM implementation for educational purpose.


## Repo Structure

```shell
.
├── ddpm.py # a DDPM class containing methods for forward diffusion (with visualization); and reverse diffusion using the UNet.
├── inference.py # a inference script with argument parser for trained models.
├── my_dataset.py # an example of making ur own dataset in pytorch.
├── scheduler.py # a beta scheduler with visualization
├── train.py # a training script with argument parser.
├── UNet.py # a minimal example of UNet with time embedding.
└── utils.py # seeding function and misc. stuffs.
```

### Train

```shell
usage: train.py [-h] -s SOURCE [-b BATCH_SIZE] [-ims IMSIZE] [-T TIMESTEPS] [-ep EPOCH] [-sd SEED] [-d DEST]

optional arguments:
  -h, --help            show this help message and exit
  -s SOURCE, --source SOURCE
                        filepath to your dataset image folder.
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        batch size.
  -ims IMSIZE, --imsize IMSIZE
                        image size.
  -T TIMESTEPS, --timesteps TIMESTEPS
                        timesteps.
  -ep EPOCH, --epoch EPOCH
                        epochs. 500 is enough to make a clear images
  -sd SEED, --seed SEED
                        seed number. Default is 42 for reproducible result
  -d DEST, --dest DEST  Destination folder path for saving results.
```


### Inference
```shell
usage: inference.py [-h] -w WEIGHT [-n NUM_SAMPLES] [-ims IMSIZE] [-T TIMESTEPS] [-sd SEED] [-d DEST]

optional arguments:
  -h, --help            show this help message and exit
  -w WEIGHT, --weight WEIGHT
                        filepath to your weight file.
  -n NUM_SAMPLES, --num_samples NUM_SAMPLES
                        num_samples.
  -ims IMSIZE, --imsize IMSIZE
                        image size.
  -T TIMESTEPS, --timesteps TIMESTEPS
                        timesteps.
  -sd SEED, --seed SEED
                        seed number. Default is 42 for reproducible result
  -d DEST, --dest DEST  Destination folder path for saving results.
```
![epoch_495](https://user-images.githubusercontent.com/77888770/205251258-acb6659b-7e92-48a4-9b6f-2ff902353b3d.png)

