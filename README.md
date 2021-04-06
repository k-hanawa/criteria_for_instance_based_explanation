# Evaluation Criteria for Instance-based Explanation

Source code for [Evaluation of Similarity-based Explanations](https://openreview.net/forum?id=9uvhpyQwzM_), which will be presented at [ICLR 2021](https://iclr.cc/Conferences/2021).


## Setup
- Download TREC dataset from [here](https://cogcomp.seas.upenn.edu/Data/QA/QC/) to `data/trec/train.txt` and `data/trec/test.txt`.


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Experiments

### Train models

For example, for MNIST with CNN, run this command to train the model:

```trai
python src/train_model.py -out results/models/mnist_cnn_0 --dataset mnist --model cnn --train_size 0.1 --seed 0 --gpu 0
```

Specify `--merge_label` when training the model to evaluate the identical subclass test.
```trai
python src/train_model.py -out results/models/merged_mnist_cnn_0 --dataset mnsit --model cnn --train_size 0.1  --seed 0 --gpu 0 --merge_label
```

For cifar10, pre-train the model with all data.
```trai
python src/train_model.py -out results/models/cifar10_cnn_all_0 --dataset cifar10 --model cnn --train_size 1 --seed 0 --gpu 0
python src/train_model.py -out results/models/cifar10_cnn_0 --dataset cifar10 --model cnn --train_size 0.1 --seed 0 --gpu 0 --pretrained_model results/models/cifar10_cnn_all_0/best_model.npz
```

#### Options
- `--dataset`
    - Name of the dataset. The following values ​​are available.
        - `mnist`
        - `cifar10`
        - `trec`
        - `cf`
- `--dataset_path`
    - Path to the directory where the dataset is stored. This option is required for `trec` and `cf`.
- `--model`
    - Name of the model. The following values ​​are available.
        - `cnn`
        - `lstm`
        - `logreg`
- `--train_size`
    - Ratio of sizes sampled from training data.
- `--merge_label`
    - If this option is specified, randomly merge labels into binary classes. Use this option when evaluating identical subclass test.
- `--seed`
    - Used for random seed.
- `--gpu`
    - GPU ID to use. Use cpu if a negative value is specified.
- `--out`
    - Path to the directory to save the trained model.

For cifar10 with MobileNetV2, use the following command instead.
```
python src/torch/train_mobilenetv2.py -out results/models/cifar10_mobilenet_0 --pretrained_model results/models/cifar10_mobilenet_all_0/best_model.pt --gpu 0
python src/torch/train_mobilenetv2.py -out results/models/cifar10_mobilenet_0 --pretrained_model results/models/cifar10_mobilenet_0/best_model.pt --traain_size 0.1 --gpu 0
```

### Evaluate each relevance metric
First, run the following command to score all training instances.

```eval
python src/identical_class_test.py --saved-dir results/models/merged_mnist_cnn_0 --dataset mnist --model cnn --test-size 500 --seed 0 --out results/all_scores/identical_class_test/score_mnist_cnn.0.pkl --gpu 0
python src/identical_subclass_test.py --saved-dir results/models/mnist_cnn_0 --dataset mnist --model cnn --test-size 500 --seed 0 --out results/all_scores/identical_subclass_test/score_mnist_cnn.0.pkl --gpu 0
```
#### Options
- `--saved-dir`
    - Path to the directory where the trained model is saved.
- `--dataset`
    - Name of the dataset. The following values ​​are available.
        - `mnist`
        - `cifar10`
        - `trec`
        - `cf`
- `--model`
    - Name of the model. The following values ​​are available.
        - `cnn`
        - `lstm`
        - `logreg`
- `--seed`
    - Used for random seed.
- `--gpu`
    - GPU ID to use. Use cpu if a negative value is specified.
- `--out`
    - Path to the pickle file to save the all score for training data.

For cifar10 with MobileNetV2, use the following command instead.
```
python src/torch/identical_class_test.py --saved-dir results/models/cifar10_mobilenet_0 --test-size 500 --seed 0 --out results/all_scores/identical_class_test/score_mnist_mobilenet.0.pkl --gpu 0
python src/torch/identical_subclass_test.py --saved-dir results/models/merged_cifar10_mobilenet_0 --test-size 500 --seed 0 --out results/all_scores/identical_subclass_test/score_mnist_mobilenet.0.pkl --gpu 0
```

Then, run the following command to calculate the success rate.

```eval
python src/eval_success_rate_identical_class.py --input results/all_scores/identical_class_test/score_mnist_cnn.0.pkl --out results/succes_rate/identical_class_test/sr_mnist_cnn.0.txt --top-k 1
python src/eval_success_rate_identical_class.py --input results/all_scores/identical_class_test/score_mnist_cnn.0.pkl --out results/succes_rate/top10_identical_class_test/sr_mnist_cnn.0.txt --top-k 10
python src/eval_success_rate_identical_subclass.py --input results/all_scores/identical_subclass_test/score_mnist_cnn.0.pkl --out results/succes_rate/identical_subclass_test/sr_mnist_cnn.0.txt --top-k 1
python src/eval_success_rate_identical_subclass.py --input results/all_scores/identical_subclass_test/score_mnist_cnn.0.pkl --out results/succes_rate/top10_identical_subclass_test/sr_mnist_cnn.0.txt --top-k 10
```
#### Options
- `--top-k`
    - Evaluate the top-k identical (sub)class test.
- `--input`
    - Path to the pickle file containing all scores.
- `--out`
    - Path to the text file to save the succes rate results. Results are tab-delimited with metrics and success rates.

### Other analysis
All figures in the paper can be found in the ju[yter notebook below:
`src/analys.ipynb`

## Citation
```
@inproceedings{hanawa2021evaluation,
  title={Evaluation of Similarity-based Explanations},
  author={Kazuaki Hanawa and Sho Yokoi and Satoshi Hara and Kentaro Inui},
  booktitle={Proceedings of the Ninth International Conference on Learning Representations},
  year={2021},
}
```
