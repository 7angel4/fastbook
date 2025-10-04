# L1
* universal approximation theorem: NNs can theoretically represent any mathematical function
    - impossible in practice, due to the limits of available data and computer hardware

# L2

## Data storage

### 1) DataLoaders
A fastai class that stores multiple `DataLoader` objects you pass to it, normally a `train` and a `valid`

```python
class DataLoaders(GetAttr):
    def __init__(self, *loaders): self.loaders = loaders
    def __getitem__(self, i): return self.loaders[i]
    train,valid = add_props(lambda i,self: self[i])
```
* 4 functionalities
    - What kinds of data we are working with
    - How to get the list of items
    - How to label these items
    - How to create the validation set


### 2) DataBlock
```python
bears = DataBlock(
    blocks=(ImageBlock, CategoryBlock),  # type of x, y
    get_items=get_image_files, # how to get a list of those file paths
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=RandomResizedCrop(128, min_scale=0.3)) 
dls = bears.dataloaders(img_path)
dls.train.show_batch(max_n=4, nrows=1, unique=True)  # show same img with diff. versions of random resized crop
```
* `Resize`: crop (default), pad, or squish to square of this size
    - e.g. `item_tfms=Resize(128, ResizeMethod.Pad, pad_mode='zeros')`


# Data Augmentation

* **Data augentation**
    - creating random variations of input data, such that they appear different, but not changing the meaning of the data
    - rotation, flipping, perspective warping, brightness changes and contrast changes
    - `RandomResizedCropt`: crop to a randomly select part of the image per epoch *(avoid losing features / wasting computations)*
    - train NNs with objects in slightly different places & sizes
    - `item_tfms`: transformations applied to a single data sample x on the CPU
    - `batch_tfms`: applied to batched data samples (individual samples collated into a mini-batch) on the GPU
        * faster, more efficient
        * e.g. `aug_transforms` for natural photo images
        * `bears = bears.new(item_tfms=Resize(128), batch_tfms=aug_transforms(mult=2))`

# Data cleaning

* `cleaner = ImageClassifierCleaner(learn)`: fastai GUI for data cleaning
    - allows you to choose a category and the training vs. validation set and view the highest-loss images (in order)
    - menus to allow images to be selected for removal or relabeling

# Problems
* out-of-domain data: data our model sees in production which is very different to training data
* domain shift: type of data that our model sees changes over time
* can never fully understand the entire behaviour of NN
* collaborative filtering: often only tell what products a user might like, not helpful recommendations


# Strengths
* DL is good at analyzing tabular data that includes natural language, or high cardinality categorical columns (containing larger number of discrete choices like zip code).

# Model => Web App

* `learn.export()`: saves a `.pkl` file containing the NN architecture + trained params + defined `DataLoaders`
* IPython widgets: JavaScript and Python combined functionalities that let us build and interact with GUI components directly in a Jupyter notebook
    - e.g. an upload button, created with `widgets.FileUpload()`
