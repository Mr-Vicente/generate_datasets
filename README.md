# Synthetic Data Generation

User friendly library to generate datasets using GANs and VAEs

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install generate_datasets.

```bash
pip install generate_datasets
```

## Usage

```python
from generate_datasets import data_access
from generate_datasets.wgan_gp_class_big import Big_WGAN

BATCH_SIZE = 64
EPOCHS = 1000
NOISE_SIZE = 128

Number_Dataset_Classes = 2

gan = Big_WGAN(BATCH_SIZE,NOISE_SIZE)
gan.load_dataset(data_access.prepare_data('gan'),Number_Dataset_Classes)
gan.train_model(EPOCHS)
gan.generate_images(10,"imgs")
```


## Library User

:Dataset:
:Iterations:

## Library Info

PGGAN \
WGAN \
IntroVae \


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Project status
On going!

## Authors and acknowledgment
Frederico Vicente & Ludwig Krippahl

## License
[MIT](https://choosealicense.com/licenses/mit/)
