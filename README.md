
<img src="images/Synth_logo.png" alt="Synthetic Data Gen Logo" title="Logo" align="left" height="60"/>

# Synthetic Data Generation

User friendly library to generate datasets using GANs and VAEs.

Paper supporting this library:
<a href="/Synthetic_Data_Generation_to_Better_Perceive_Classifier_Predictions.pdf" class="image fit">Synthetic Data Generation To Better Perceive Classifier Predictions</a>

## Installation

Use git to install generate_datasets.

```bash
git clone https://github.com/Mr-Vicente/generate_datasets.git
```

## Usage

### GAN

```python
from generate_datasets import data_access
from generate_datasets.WGAN_GP.wgan_gp_class_generic import WGAN_GP

EPOCHS = 500
params = {
    'lr': 0.0001,
    'beta1': 0,
    'batch_size': 64,
    'latent_dim': 128,
    'image_size': 152

}

class_names = ['Blond-yellow','Yellow','Orange','Orange-Brown','Blond',
                'Light-Brown','Brown','Black','Gray','White']


Number_Dataset_Classes = len(class_names)

gan = WGAN_GP(params)
gan.load_dataset(data_access.prepare_data('gan'),Number_Dataset_Classes)
gan.train_model(EPOCHS)
gan.generate_images(10,"imgs")
```

If no classifier is provided to GANs, the gan model will behave like a normal gan.
The interest of one providing a classifier to our GAN model might be to better
understand how it is operating, or even to generate images of a specific class.

### VAE

```python
from generate_datasets import data_access
from generate_datasets.VAE.generic_vannila_vae import VAE
from generate_datasets.Processing import process_cartoon

EPOCHS = 500
n_images = 10
OUTPUT_DIR = "imgs_gen"

params = {
    'lr': 0.0001,
    'beta1': 0,
    'batch_size': 64,
    'latent_dim': 128,
    'image_size': 152

}

class_names = ['Blond-yellow','Yellow','Orange','Orange-Brown','Blond',
                'Light-Brown','Brown','Black','Gray','White']

def load_cartoon_data():
    images, labels = process_cartoon.decode_data_cartoon()
    return images, labels

tf.keras.backend.clear_session()

model = VAE()
model.load_dataset(data_access.prepare_dataset('vae',load_cartoon_data(),image_size=(128,128)))
model.train_model(epochs)
model.generate_images(n_images,OUTPUT_DIR)

```

## Library Info

This library has several generative models at your despose:

* GAN (Generative Adversarial Network)
    * Status: Working
    * Paper: [GAN](https://arxiv.org/abs/1406.2661)
    * Official Implementation: None
* VAE (Variational Autoencoder)
    * Status: Working
    * Paper: [VAE](https://arxiv.org/abs/1312.6114)
    * Official Implementation: None
* WGAN (Wasserstein GAN)
    * Status: Working
    * Paper: [WGAN](https://arxiv.org/abs/1701.07875)
    * Official Implementation: None
* WGAN-GP (Wasserstein GAN with Gradient Penalty)
    * Status: Working
    * Paper: [WGAN-GP](https://arxiv.org/abs/1704.00028)
    * Official Implementation: None, but check this one [WGAN-GP Implementation](https://github.com/igul222/improved_wgan_training)
* PGGAN (Progressive Growing GAN)
    * Status: Not working
    * Paper: [PGGAN](https://arxiv.org/abs/1710.10196)
    * Official Implementation: [PGGAN Implementation](https://github.com/tkarras/progressive_growing_of_gans)
* IntroVAE (Introspective Variational Autoencoder)
    * Status: Not giving proper results
    * Paper: [IntroVAE](https://arxiv.org/abs/1807.06358)
    * Official Implementation: None

To learn more about these Generative models visit the referenced papers/implementations.


## Generated Images (examples)

Epochs|Datasets|Original Images|WGAN-GP | VAE | IntroVAE
------|--------|---------------|--------|--------|------
200|Fashion MNIST|<img alt="Fashion MNIST image sample" src="images/Fashion_Mnist_Dataset_Image.png" width="64"> |![Image generated by WGAN-GP](images/WGAN-GP/train_1.png?raw=true "WGAN-GP Image") | ![Image generated by WGAN-GP](images/WGAN-GP/train_1.png?raw=true "WGAN-GP Image") | ![Image generated by WGAN-GP](images/WGAN-GP/train_1.png?raw=true "WGAN-GP Image")| -
200|Cartoon Dataset|<img alt="Cartoon Dataset image sample" src="images/Cartoon_Dataset_Image.png" width="64"> |![Image generated by WGAN-GP](images/WGAN-GP/train_1.png?raw=true "WGAN-GP Image") | ![Image generated by WGAN-GP](images/WGAN-GP/train_1.png?raw=true "WGAN-GP Image") | ![Image generated by WGAN-GP](images/WGAN-GP/train_1.png?raw=true "WGAN-GP Image")| -
200|Train Dataset|<img alt="Train Dataset image sample" src="images/Train_Dataset_Image.png" width="64">|![Image generated by WGAN-GP](images/WGAN-GP/train_1.png?raw=true "WGAN-GP Image") | ![Image generated by WGAN-GP](images/WGAN-GP/train_1.png?raw=true "WGAN-GP Image") | ![Image generated by WGAN-GP](images/WGAN-GP/train_1.png?raw=true "WGAN-GP Image")| <img alt="Image generated by IntroVae" src="images/IntroVAE/IntroVAE_Train.png" width="64">
200|LSUN Bedroom Dataset|<img alt="LSUN Bedroom Dataset image sample" src="images/LSUN_DATASET_IMAGE.png" width="64">|<img alt="Image generated by WGAN-GP" src="images/WGAN-GP/LSUN_WGAN-GP.png" width="64">| ![Image generated by WGAN-GP](images/WGAN-GP/train_1.png?raw=true "WGAN-GP Image") | ![Image generated by WGAN-GP](images/WGAN-GP/train_1.png?raw=true "WGAN-GP Image")| -


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Project status
On going!

## Authors and acknowledgment
Frederico Vicente & Ludwig Krippahl

## License
[MIT](https://choosealicense.com/licenses/mit/)
