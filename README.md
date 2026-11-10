# PCA, LDA & SVD for Face Image Analysis and Movie Recommendation

**DSL502 Mathematics Assignment**\
Manual implementation of Principal Component Analysis (PCA), Linear
Discriminant Analysis (LDA), and Singular Value Decomposition (SVD)
using Python and NumPy.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![NumPy](https://img.shields.io/badge/NumPy-manual%20linear%20algebra-013243)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange)
![LaTeX](https://img.shields.io/badge/Report-LaTeX-008080)

This repository contains the implementation notebook, experiment
outputs, dataset preparation workflow, and LaTeX report for four
experiments:

1.  **PCA:** face-image analysis and reconstruction under two
    illumination conditions.
2.  **PCA + LDA:** supervised subject-wise discrimination.
3.  **SVD:** low-rank image reconstruction and denoising.
4.  **SVD-based recommendation:** collaborative filtering using
    MovieLens 100K.

> **Implementation constraint:** Core PCA, LDA, and SVD calculations are
> implemented using matrix operations and symmetric eigenvalue
> decomposition (`numpy.linalg.eigh`). Built-in PCA/LDA/SVD estimators
> are not used for the core calculations.

------------------------------------------------------------------------

## Repository structure

``` text
dsl502-assignment/
├── .github/
│   └── workflows/
│       └── latex.yml
├── data/
│   ├── raw/
│   │   ├── PCA_PIE_Data/
│   │   │   ├── lights/
│   │   │   └── nolights/
│   │   └── ml-100k/
│   └── processed/
│       └── pie_dataset.npz
├── notebooks/
│   └── assignment_01.ipynb
├── outputs/
│   └── experiment plots
├── figures/
│   └── figures referenced by the report
├── Maths_Assignment_Report.tex
├── README.md
└── pyproject.toml
```

The diagram shows the intended organization. Actual generated files may
vary. Raw datasets are downloaded separately and do not need to be
committed to GitHub.

------------------------------------------------------------------------

## Experiments and key findings

### Q1 --- Principal Component Analysis (PCA)

The CMU PIE subset contains five subjects: `Person1`, `Person2`,
`Person9`, `Person19`, and `Person46`. Each subject has 21 images per
illumination condition. Images are grayscale and have a resolution of
`100 × 100` pixels.

For each condition:

-   Number of images: 105
-   Number of features per image: 10,000
-   Data matrix: `105 × 10000`

PCA is computed separately for `lights` and `nolights` using the dual
Gram matrix:

$$ G = \frac{1}{N-1} X_c X_c^T $$


**Findings**

-   PC1--PC3 captures dominant facial structure and broad intensity
    variations.
-   PC5--PC9 contains lower-variance patterns and cannot independently
    preserve the dominant structure.
-   PC1--PC9 gives the best numerical reconstruction among the tested
    component groups.
-   The `nolights` condition has a larger share of variance concentrated
    in PC1 than `lights`.
-   PCA is unsupervised; its components can capture illumination
    variation as well as facial variation.

### Q2 --- Linear Discriminant Analysis (LDA)

The original 10,000-dimensional images are first reduced to 30 PCA
features. LDA is then applied to the reduced features. With five subject
classes, the maximum number of meaningful discriminant directions is:

$$ C-1=5-1=4 $$

**Fisher separation ratio**

  Illumination condition     PCA (PC1--PC2)   LDA (LD1--LD2)
  ------------------------ ---------------- ----------------
  Lights                             0.5142         622.1718
  NoLights                           0.2563         270.6606

Ten combinations of three subjects were also evaluated.

-   Highest Lights ratio: **2034.61**, for Person9 + Person19 +
    Person46.
-   Highest NoLights ratio: **836.74**, for Person2 + Person9 +
    Person46.

These values are two-dimensional Fisher separation ratios, not
classification accuracies.

### Q3 --- SVD image denoising

Three PIE face images were selected:

-   Person1 --- Flash01
-   Person9 --- Flash01
-   Person46 --- Flash01

Gaussian and Salt-and-Pepper noise were added. Reconstructions were
compared using the top 5, 20, and 50 singular values. Gaussian noise
levels of (`\sigma=10`{=tex},25,50) were also tested.

**Average MSE at the selected noise setting**

  Noise type          Noisy image    Top 5   Top 20    Top 50
  ----------------- ------------- -------- -------- ---------
  Gaussian                 542.60   247.71   298.76    478.67
  Salt-and-Pepper         1105.96   324.00   649.82   1001.69

For Gaussian noise:

-   At (`\sigma=10`{=tex}), rank 20 gave the lowest average MSE:
    **60.17**.
-   At (`\sigma=25`{=tex}), rank 5 performed best: approximately
    **252.54**.
-   At (`\sigma=50`{=tex}), rank 5 performed best: approximately
    **472.40**.

Low-rank approximation suppresses some noise but can also remove facial
detail. The lowest MSE does not necessarily imply the best perceptual
reconstruction.

### Q4 --- MovieLens 100K recommendation

The MovieLens 100K dataset contains:

-   100,000 ratings
-   943 users
-   1,682 movies
-   Rating scale: 1 to 5

A user--movie matrix of shape `943 × 1682` was constructed. The data was
divided into 80,000 training ratings and 20,000 test ratings. User means
were calculated using training ratings only.

The centered matrix is decomposed using the smaller Gram matrix:

$$ G=R_cR_c\^T $$

**Test-set prediction performance**

    Components ((k))            MAE           RMSE
  ------------------ -------------- --------------
                   5       0.784795       0.991852
                  10   **0.780559**   **0.987493**
                  20       0.783247       0.991055
                  50       0.795900       1.005334
                 100       0.811093       1.021614

The best tested prediction performance was obtained at **(k=10)**.
Approximately 321 components were required to retain 90% of the squared
singular-value energy. Energy retention and held-out prediction accuracy
are different objectives.

Top-10 recommendations were generated for five selected users.

------------------------------------------------------------------------

## Technology stack

  Tool              Purpose
  ----------------- ------------------------------------------------
  Python 3.13       Programming language
  NumPy             Matrix operations and eigenvalue decomposition
  Pandas            Dataset loading and tabular processing
  Matplotlib        Visualizations and plots
  Pillow            Face-image loading and processing
  JupyterLab        Interactive implementation notebook
  uv                Python environment and dependency management
  LaTeX / latexmk   Technical report compilation

The project was developed in Ubuntu under WSL and can also run in a
compatible Linux environment.

------------------------------------------------------------------------

## Setup

### 1. Clone the repository

``` bash
git clone https://github.com/ltd-ARYAN-pvt/DSL502-PCA-LDA-SVD.git
cd DSL502-PCA-LDA-SVD
```

Replace the placeholder with the URL of your GitHub repository.

### 2. Install uv

Follow the official installation instructions:

<https://docs.astral.sh/uv/getting-started/installation/>

### 3. Install project dependencies

``` bash
uv sync
```

The core Python dependencies are:

``` bash
uv add numpy pandas matplotlib pillow jupyterlab ipykernel
```

### 4. Launch JupyterLab

``` bash
uv run jupyter lab
```

Open `notebooks/assignment_01.ipynb` and execute the notebook cells in
order.

------------------------------------------------------------------------

## Dataset setup

Datasets are downloaded separately and placed under `data/raw/`.

### CMU PIE subset

Expected directory structure:

``` text
data/raw/PCA_PIE_Data/
├── lights/
│   ├── Person1/
│   ├── Person2/
│   ├── Person9/
│   ├── Person19/
│   └── Person46/
└── nolights/
    ├── Person1/
    ├── Person2/
    ├── Person9/
    ├── Person19/
    └── Person46/
```

Each subject directory is expected to contain 21 grayscale images
(`flash_01.jpg` through `flash_21.jpg`).

The preprocessing workflow creates:

``` text
data/processed/pie_dataset.npz
```

This NumPy archive stores the `X_lights` and `X_nolights` arrays.

### MovieLens 100K

Download the dataset from the official GroupLens page:

<https://grouplens.org/datasets/movielens/100k/>

Direct archive:

<https://files.grouplens.org/datasets/movielens/ml-100k.zip>

Extract it under:

``` text
data/raw/ml-100k/
```

Files used by the notebook:

  File       Description
  ---------- ------------------------------------------
  `u.data`   User ID, movie ID, rating, and timestamp
  `u.item`   Movie metadata, including movie titles

Follow the dataset's citation and usage guidance. Do not commit raw
datasets unless redistribution is permitted and appropriate.

------------------------------------------------------------------------

## Manual numerical methods

The core implementations use:

-   `numpy.linalg.eigh` for symmetric eigenvalue and eigenvector
    calculations.
-   Matrix multiplication to construct Gram matrices and scatter
    matrices.
-   Eigenvalues and eigenvectors to recover principal directions and
    singular vectors.
-   Explicit low-rank reconstruction for selected component counts.

The core experiments should not be replaced with `numpy.linalg.svd`,
SciPy SVD/SVDS, scikit-learn PCA/LDA, or TruncatedSVD because these
built-in decomposition routines are outside the assignment's
implementation constraint.

------------------------------------------------------------------------

## LaTeX report 

### Compile locally in WSL Ubuntu

Install the LaTeX toolchain:

``` bash
sudo apt update
sudo apt install -y latexmk texlive-xetex texlive-latex-extra texlive-fonts-recommended
```

From the repository root, run:

``` bash
latexmk -xelatex -interaction=nonstopmode -halt-on-error Maths_Assignment_Report.tex
```

The generated PDF will be:

``` text
Maths_Assignment_Report.pdf
```

To remove auxiliary build files:

``` bash
latexmk -c
```

### Compile using GitHub Actions

1.  Commit and push the `.tex` report, the `figures/` directory, and
    `.github/workflows/latex.yml`.
2.  Open the repository on GitHub.
3.  Select the **Actions** tab.
4.  Open the **Build LaTeX report** workflow run.
5.  Download the `assignment-report-pdf` artifact.

The generated PDF is stored as a workflow artifact. It is not
automatically committed to the repository. To make it a normal
repository file, commit the PDF separately or configure a publishing
workflow.

### Report figure paths

The LaTeX source references figures under `figures/`. If the notebook
saves plots under `outputs/`, copy the required images into `figures/`
or update the image paths in the `.tex` file.

The figure filenames in the report source and repository must match.
Missing figure files can cause compilation to fail.

------------------------------------------------------------------------

## Reproducibility and evaluation notes

-   The train/test split uses NumPy random seed `42`.
-   For exact noise-image reproducibility, set a fixed random seed
    before generating noise and reuse the same noisy arrays for
    comparisons.
-   MAE and RMSE are calculated on held-out test ratings.
-   Recommendation generation should exclude movies rated in the
    training set. Held-out test ratings should not be used to decide
    which movies are already known to a user.
-   The reported best (k) is the best among the tested values, not a
    claim of global optimality.

------------------------------------------------------------------------

## References

1.  Sim, T., Baker, S., & Bsat, M. (2001). *The CMU Pose, Illumination,
    and Expression (PIE) Database*. Carnegie Mellon University, Robotics
    Institute Technical Report CMU-RI-TR-01-02.
2.  Harper, F. M., & Konstan, J. A. (2015). The MovieLens Datasets:
    History and Context. *ACM Transactions on Interactive Intelligent
    Systems*, 5(4), Article 19.
3.  GroupLens Research. [MovieLens 100K
    Dataset](https://grouplens.org/datasets/movielens/100k/).
4.  NumPy documentation: <https://numpy.org/doc/>.

------------------------------------------------------------------------

## Author

-   **Student: Aryan Pandey**
-   **Course: DSL502**
-   **Institute: IIT-BH**
