# Cell Counter

Processes microscopy images to count cells and identify which are pyknotic. This is internal/development-grade source code, provided as-is primarily for reference purposes. However, we hope this program will help other labs establish an automated method of analyzing histologic images, and we hope someone will create a more robust version of this program.  

Please cite our work if you use this code: <https://doi.org/10.1002/btm2.70058>


# Installation Instructions

This program is written entirely in the Python programming language, version 3.10.11. Please ensure Python version 3.10 is downloaded. This program was developed in MacOS (Sequoia 15.3.2) and has also been tested on the Ubuntu Linux distribution, version 22.04. Theoretically, it should not have trouble running on Windows OS with appropriate configuration.

These directions are intended to help novice programmers get started. In a terminal, `cd` into this directory, likely: `cd cell_counter`. Execute these commands in a terminal window, written for Linux/MacOS. This program was developed and tested with MacOS Adjustments may be needed for Windows and other operating systems.

1. Create a virtual environment: `python3.10 -m venv .venv`

1. Activate virtual environment: `source .venv/bin/activate` (Windows PowerShell: `source .\.venv\Scripts\Activate.psl`)

1. Upgrade pip and install dependencies: `pip install --upgrade pip && pip install -r requirements.txt`. 

Congratulations, you have installed `cell_counter` on your computer!


# Using `cell_counter`

Microscopy images may be viewed by running the command `python image_viewer.py [FILE PATHS]`

Cell counts can be performed with the command `python cell_counter.py [-OPTIONS] [SOURCE] [OUTPUT DIRECTORY]`


# Usage instructions and helpful information

Pre-processing steps are used to refine each image. These steps first reduce image artefact and then create a mask of the image, indicating which regions may contain cells. We recommend using "image_viewer.py" to ensure the current pre-processing are sufficient for your application. Preprocessing steps may be modified within the code.  

By default, cell candidates are detected in the pre-processed and masked image using the Laplacian of the Gaussian method. This searches the provided image for roughly circular "blobs" and labels each as a cell candidate. The center location and the "sigma" value for the gaussian that best fit each blob (cell/nuclei) is recorded. This method can tolerate overlapping cells, although the extent of overlap permitted may be tuned for your specific application. Subsequent filtering steps remove false positives.  

By default, a pre-defined cluster analysis is available to classify DAPI-stained cell nuclei as pyknotic or non-pyknotic. This will likely need to be tuned or changed depending on your specific application. Cells may be classified by explicitly setting thresholds or by providing your own data to develop a new cluster-based model.  

Even if nuclei are not accurately labelled as pyknotic/non-pyknotic (if this labelling matters for your application), properties for all detected cells can be saved and accessed in a spreadsheet after analyzing a batch of images. These properties are intended to aid in developing your own classification scheme.


## Developing a new classifier

Users are encouraged to use a pre-trained classifier if possible and appropriate for their situation.

If a new classifier must be trained, a representative sample of images (recommend at least 50-100) should be manually annotated as described below.

Steps for developing a new classifer:

1. Set paths and run `python cell_counter.py` without a random forest classifier to count cells and save their properties. Classifications will be very inaccurate, but the classifications do not impact cell property calculations. The output file `properties.csv` will be important for the next step.

1. Set paths/constants and run `python train_rdf_model.py`, providing paths to folders containing identically named `.nd2` and `.png` raw microscopy and annotated image files, respectively. This will create a new `properties_updated.csv` file where the "known_pyknotic" column has a value of 0/1 for cells in images that were annotated. A random forest classifier will be trained. Stop here if the random forest classifier already performs sufficiently well - set the principal component thresholds in `train_rdf_model.py` to None to only use the random forest classifier.

1. To improve the classifier's performance, set paths and run `python fit_pca.py` to perform principal component analysis on the `properties_updated.csv` file. This will create a graph of the first 2 principal components and save a file called `pca_transformation.xlsx` with data to reproduce the same transformation with other sets of images.

1. Use the data in `pca_transformation.xlsx` to identify an acceptable domain in the PCA-derived latent space for all cells (optional but recommended). Currently, the code is setup to only consider the first two principal components. A subdomain should then be defined where cells are likely to be pyknotic. In the `cell_counter.py` file, define the domain of "pyknotic candidates" in this latent space.

1. Rerun `python cell_counter.py`. The cells classified as pyknotic, especially in `properties.csv`, reflect the previously defined "pyknotic candidates" that will undergo further consideration by the random forest classifer.

1. Make a copy of `properties.csv` called `training_properties.csv`. Without changing the order of the `training_properties.csv` file, remove all rows where "classified_pyknotic" (last column) is "FALSE". 
    1. I recommend creating a column of row indices and sorting by the "classified_pyknotic" column and then by original row index to delete all non-candidates in one large block. Then delete the column of row indices you created.

1. Now, run `python train_rdf_model.py` to train a new random forest model that specifically classifies the previously defined pyknotic candidates as pyknotic or non-pyknotic. All other cells are known to be non-pyknotic, helping to improve the model's performance.

1. For future classifications, set `cell_counter.py` to reference the same `pca_transformation.xlsx` file you previously made, use the same principal component thresholds as before, and now reference the new random forest classifier `.pkl` file developed in the previous step. You now have a new classification scheme.


### Manually annotating pyknotic/target nuclei

Target nuclei may be annotated in ImageJ using the following protocol.

1. Use multipoint tool to identify pyknotic cells. Click as close to the center of the cell as possible.

1. Double click on the multipoint tool icon to open its configurations.
    1. Type: dot  

    1. Color: yellow  

    1. Size: small  

    1. Label points: off

1. Image > overlay > Add selection

1. Image > overlay > Flatten

1. File > Save as > PNG

Save the `.png` files in a different folder. Aside from the extension, ensure the corresponding `.png` and `.nd2` files have the same name.  
