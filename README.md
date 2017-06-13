# Image caption generator using Chainer

# Requirement

* [Chainer 2.0.0](http://chainer.org/)
* [Cupy 1.0.0](http://docs.cupy.chainer.org/en/stable/)
* [Pillow](https://pypi.python.org/pypi/Pillow/)

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
* -s, --sentence: (required) sentence dataset file path.
* -i, --image: (required) image feature file path.
* -m, --model: (optional) input model file path without extension.
* -o, --output: (required) output model file path without extension.
* --iter: (optional) the number of iterations (default: 100).

## Convert VGG 19-layer caffemodel to pkl file

```
$ python src/convert_caffemodel_to_pkl.py VGG_ILSVRC_19_layers.caffemodel vgg19.pkl
```

Parameters:
* caffe model file path.
* output pkl file path.

## Generate image caption

```
$ python src/generate_caption.py -s dataset.pkl -i vgg19.pkl -m model/caption_gen_0010.model -l image/list.txt
```

Options:
* -s, sentence: (required) sentence dataset file path.
* -i, --image: (required) image model file path.
* -m, --model: (required) trained model file path.
* -l, --list: (required) image path list file.

### Image path list file sample

```
image/asakusa.jpg
image/tree.jpg
```

# License

MIT License
