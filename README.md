# VaCe: Valid Centers for Chromosomes for Cancer Research: Upgrading Chromosome Analysis
This contains the code for our chromosome segmentation tool. 

## Directory

| File      | Description |
| ----------- | ----------- |
| config.ini      | Configuration file to set parameters       |
| environment.yml   | File to create the environment        |



| Folder      | Description |
| ----------- | ----------- |
| sample      | Folder containing sample images to use with our project       |
| src   | Contains our Python scripts        |

## Installation
To get started, run the following code to create the environment. Run this any time you want to use the tool!

```
git clone 
cd VaCe
conda env create -f environment.yml
conda activate vace
```

## Image Specifications
Input folder will only read .tif files

## Tasks
### `make centers`
Add description

Set parameters in config.ini under `Centers`:

````
image_path : path to folder containing images
````

#### Output

1. **norm folder** - folder containing normalized images
add output