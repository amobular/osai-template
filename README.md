# ⚡ THOR stack template
This project contains a general template for AI research and development. We make use of the following libraries":
- Torch (with Lightning for better ML code organisation)
- Hydra (for configuration management)
- Optuna (for hyperparameter searching and running experiments)
- Ray (for parallel processing and larger data tasks)

We decided to call this ML stack 'THOR'.

The template contains basic code, that will be needed for (almost) every ML project. This includes:
- Basic Torch model implementation
- Basic training loop and metric logging
- Extendable code, that very easily allows adding new models, datasets, and configuring them
- Common utility code, like setting up weight-decay
- Easy dataset downloading and preparation for Kaggle competitions

For simple instructions on how to use this template, refer to the tutorials in the `docs` directory.
