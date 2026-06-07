# Structured Distillation for Few-Shot Heterogeneous Federated Unlearning

## Architecture diagram：

---

![1780816955540](C:\Users\12197\AppData\Roaming\Typora\typora-user-images\1780816955540.png)

## **Running**：

---

### 1、requirement install：

```
pip install -r requirements.txt
```

The project is built with PyTorch 1.7.0. Key dependencies include `torch`, `yacs` (configuration), `wandb` (experiment tracking), `timm` (ViT models), and `tensorboardX`.

### 2、Datasets

Download the datasets and place them in the `root/mdistiller/dataset/data/` directory under the project root.

|   Dataset    |                        Download links                        |
| :----------: | :----------------------------------------------------------: |
| TinyImageNet |       http://cs231n.stanford.edu/tiny-imagenet-200.zip       |
|   Food101    |      http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz       |
|  OxfordPets  | https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz |
|   BreakHis   |   http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz    |

### 3、 Structured Alignment Federated Learning (SAFL)

Run the federated learning with hierarchical structural distillation:

```bash
python tools/fed_train.py --cfg configs/<dataset>/fed/<config>.yaml
```

Configuration files for SAFL training are in `configs/<dataset>/`(IID and Non-IID). Key parameters include client/server architectures, number of communication rounds, local epochs, learning rates, and non-iid alpha.

### 4、 Structured Perturbation Unlearning (SPU)

Run the structured perturbation unlearning:

```
python tools/train_unlearning.py --cfg configs/<dataset>/unlearning_test/<config>.yaml
```

Configuration files for unlearning are in `configs/<dataset>/`. Key parameters include target clients to unlearn, number of recovery rounds, backdoor attack settings, and perturbation optimizer hyperparameters.



