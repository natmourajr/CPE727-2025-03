# Double Descent

Model specifications in PyTorch are in [models](models). 
Example notebooks for plotting are presented.

## Full List of Experiments (the Data in the links)

#### Model DD

|Dataset| Architecture  | Optimizer | Comments |Data Link|
|:---:|:---:|:---:|:---:|:---:|
| Cifar10 with 0, 10, 20j% label noise and data-aug| ResNet18| Adam with LR=0.0001| |[link](https://console.cloud.google.com/storage/browser/hml-public/dd/cifar10-resnet18k-50k-adam?project=ml-theory&folder=true&organizationId=658616372004)|
| Cifar10 with 0, 10, 20% label noise and data-aug| MCNN| SGD with LR $`\propto 1/\sqrt{T}`$ | |[link](https://console.cloud.google.com/storage/browser/hml-public/dd/cifar10-mcnn-p10-sgd?project=ml-theory&folder=true&organizationId=658616372004), [link](https://console.cloud.google.com/storage/browser/hml-public/dd/cifar10-mcnn-p20-sgd-rep2?project=ml-theory&folder=true&organizationId=658616372004), [link](https://console.cloud.google.com/storage/browser/hml-public/dd/cifar10-mcnn-p20-sgd-rep3?project=ml-theory&folder=true&organizationId=658616372004), [link](https://console.cloud.google.com/storage/browser/hml-public/dd/cifar10-mcnn-p20-sgd-rep4?project=ml-theory&folder=true&organizationId=658616372004), [link](https://console.cloud.google.com/storage/browser/hml-public/dd/cifar10-mcnn-p20-sgd-rep5?project=ml-theory&folder=true&organizationId=658616372004), [link](https://console.cloud.google.com/storage/browser/hml-public/dd/cifar10-mcnn-p10-sgd?project=ml-theory&folder=true&organizationId=658616372004)|
| Cifar10 with 0, 10, 20% label noise and **no data-aug** | MCNN| SGD with LR $`\propto 1/\sqrt{T}`$ | |[link](https://console.cloud.google.com/storage/browser/hml-public/dd/cifar10-mcnn-noaug-sgd?project=ml-theory&folder=true&organizationId=658616372004)|
| Cifar100 with 0, 10, 20% label noise and data-aug| ResNet18| Adam  with LR=0.0001 | |[link](https://console.cloud.google.com/storage/browser/hml-public/dd/cifar100-resnet18k-50k-adam)|
| Cifar100 with 0, 10, 20% label noise and data-aug| MCNN| SGD with LR $`\propto 1/\sqrt{T}`$ | |[link](https://console.cloud.google.com/storage/browser/hml-public/dd/pct-cifar100-mcnn-p0-sgd-noaug-reps)|
| IWSTL'14| Transformers| SGD with warmup then LR $`\propto 1/\sqrt{T}`$ | |[link](https://console.cloud.google.com/storage/browser/hml-public/dd/nlp-model-dd/)|
| WMT'14| Transformers| SGD with warmup then LR $`\propto 1/\sqrt{T}`$ | |[link](https://console.cloud.google.com/storage/browser/hml-public/dd/nlp-model-dd/)|


#### Sample DD
|Dataset| Architecture  | Optimizer | Comments |Data Link|
|:---:|:---:|:---:|:---:|:---:|
| IWSTL'14| Transformers| SGD with warmup then LR $`\propto 1/\sqrt{T}`$ |  |[link](https://console.cloud.google.com/storage/browser/hml-public/dd/nlp-sample-dd)|
| Cifar10 with 20% label noise and varying data size and data-aug | MCNN| SGD with LR $`\propto 1/\sqrt{T}`$ | |[link](https://console.cloud.google.com/storage/browser/hml-public/dd/cifar10-mcnn-p10-sgd), [link](https://console.cloud.google.com/storage/browser/hml-public/dd/cifar10-mcnn-p20-sgd)|
#### Epoch DD
|Dataset| Architecture  | Optimizer | Comments |Data Link|
|:---:|:---:|:---:|:---:|:---:|
| Cifar10 with 0-20% label noise and data-aug|  ResNet18| Adam with LR=0.0001 |  |[link](https://console.cloud.google.com/storage/browser/preetum/data/cifar10_mcnn/?project=ml-theory&authuser=2)|
| Cifar100 with 0-20% label noise and data-aug|  ResNet18| Adam with LR=0.0001 |  |[link](https://console.cloud.google.com/storage/browser/preetum/data/cifar10_mcnn/?project=ml-theory&authuser=2)|
| Cifar10 with 0-20% label noise and data-aug|   MCNN| SGD with LR $`\propto 1/\sqrt{T}`$  |  |[link](https://console.cloud.google.com/storage/browser/preetum/data/cifar10_mcnn/?project=ml-theory&authuser=2)|
