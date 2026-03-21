### Info

Write short, clear, simple code. Achieve only whats necessary. Do not over engineer. Think carefully planning out steps. Complete each stage and wait for approval before moving to the next.

You need to create a image segmentation model that segments fillaments in yeast cells.

Test dataset: data/annotated_data.pkl
Train dataset: synthetic_data/

These images are grayscale and have a 128x128 resolution

For testing model strategies you are running on an M5 macbook. I do have access to an HPC GPU cluster for later training - if you think it's necessary to move to a HPC GPU at any stage say so.

Inputs:

- a 2D image from a tiff file, should pass to the model as an ndarray

Outputs:

- 2D boolean mask. All elements false if no filament present.

### Stage 1:

Investigate which objective functions will work best and make the most sense for the dataset with the segmentation objective.

### Stage 2

Data cleaning. Investigate the dataset consider different strategies to normalize and clean it (Noise removal?).

### Stage 3:

Planning. Which model architectures will work best? Consider using pre-implemented, accessible open source models were possible.

### Stage 4:

Basic testing. Test a variety of strategies to a small level of depth.

### Stage 5:

Hyper parameter tuning
