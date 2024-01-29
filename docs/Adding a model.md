# Adding a model
Adding a model with this template is very simple. This tutorial explains how to add code for your (PyTorch) model;
register your model for use; and how to add configurations for your model.

## Basic use
### Step 1: Define your model
Create a `.py` file in `src/model`. FOr this example, we are naming the file `my_model.py`.
After this, implement your PtTorch model, we will use a very basic example here:
```python
import torch
from torch.nn import Module, Parameter

class MyModel(Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param = Parameter(torch.rand(10))

    def forward(self, x):
        return x + self.param
```

### Step 2: Register your model
You have two options to register your model for easy use: you can manually implement it in `src/model/common.py`, or
you can use the `@register_model` decorator. The first on is done by adding the following code in `src/model/common.py`:

```python
# Import your model:
...
from src.model.my_model import MyModel
...
def get_model(model_cfg: ModelConfig) -> Module:
    # These two lines register your model under the name 'my-model'.
    # Note: leave the rest of the function untouched.
    if model_cfg.name in ["my-model"]:
        return MyModel(**model_cfg.params)
    ...
```
And then, your model is registered under the namespace `my-model`! If you do not need to add a lot of customization
when initializing your model, you can also easily register it using the `@register_model` decorator. This is done by
changing your `my_model.py` file to the following:
```python
import torch
from torch.nn import Module, Parameter

from . import register_model

@register_model("my_model")
class MyModel(Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param = Parameter(torch.rand(10))

    def forward(self, x):
        return x + self.param
```
Your model is now registered under the namespace `my-model`! You can leave `common.py` untouched.

### Step 3: Adding configuration for your model
The last step for basic use is to add a simple configuration for your model. Make a file in `run/conf/module/model`, and
name it `my_model.yaml`. This YAML file should look like:
```yaml
name: "my_model"
params: {}
```
Note that the `name: "my_model"` here refers to the namespace under which you registered your model. If you want to
make use of this configuration, you will have to change `run/conf/module/default.yaml` (or whatever YAML file is in use
for the wrapper). Change the `defaults` to the following:
```yaml
defaults:
  ...
  - model: my_model
  ...
```
Here, `my_model` refers to the name of the configuration file you created, excluding the `.yaml` suffix. After you did
this, the class `MyModel` will be initialized and used for any training, inference, etc.

## Adding parameters to the configuration
Adding parameters is really easy and done in two steps:
### Step 1: Add parameters to your model's class
In this case, we will just add a simple string parameter and print it upon initialization:
```python
import torch
from torch.nn import Module, Parameter

from . import register_model

@register_model("my_model")
class MyModel(Module):
    def __init__(self, *args, some_param: str, **kwargs):
        super().__init__(*args, **kwargs)
        print(f"Received the parameter: {some_param}")
        self.param = Parameter(torch.rand(10))

    def forward(self, x):
        return x + self.param
```

### Step 2: Update your configuration
Add the parameter to your configuration file:
```yaml
name: "my_model"
params:
  some_param: "Hello, World!"
```
And it's done! Now, if you run `run/train`, for example, you will see `Received the parameter: Hello, World!` in your
stdout.

## Tips for usage

### Multiple configurations
A very good perk of this system is that you can very easily try a lot of different configurations for the same model.
For example, if you have the following model:
```python
@register_model(["neuralnet_small", "neuralnet_large"])
class NeuralNetworkModel(Module):
    def __init__(self, *args, num_layers: int, neurons_per_layer: int, **kwargs):
        super().__init__(*args, **kwargs)
        # Use 'num_layers', and 'neurons_per_layer' to initialize some neural network model.
        ...

    def forward(self, x):
        ...
```
Notice here that the model is being registered under multiple namespaces: `neuralnet-small` and `neuralnet-large`. You
can now make multiple YAML configurations:
```yaml
# neuralnet_small.yaml
name: "neuralnet_small"
params:
  num_layers: 4
  neurons_per_layer: 16
```
```yaml
# neuralnet_large.yaml
name: "neuralnet_large"
params:
  num_layers: 32
  neurons_per_layer: 128
```
And now, you can very easily switch between these two model configurations by changing the module configuration's model
default between `neuralnet_small` and `neuralnet_large`. This makes it easy to implement many different configurations
for a search, using the same model class.
