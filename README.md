# Image caption generator using Chainer

# Requirement

* [Chainer 1.5.1](http://chainer.org/)

# Usage

## Download dataset

1. Download flickr8k.zip, flickr30k.zip, or coco.zip from:
http://cs.stanford.edu/people/karpathy/deepimagesent/
1. Extract downloaded zip file, and you'll get dataset.json and vgg_feats.mat.

## Convert dataset

```
$ python src/convert_dataset.py dataset.json dataset.pkl
```

Parameters:
* sentence JSON file of dataset.
* output pkl file.

## Train dataset

```
$ python src/train.py -g 0 -s dataset.pkl -i vgg_feats.mat -o model/caption_gen
```

Options:
* -g, --gpu: (optional) GPU device index (default: -1).
* -s, --sentence: (required) sentence dataset file.
* -i, --image: (required) image feature file.
* -m, --model: (optional) input model file path without extension.
* -o, --output: (required) output model file path without extension.
* --iter: (optional) the number of iterations (default: 100).

## License

MIT License
