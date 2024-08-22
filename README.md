# CAMELS Multifield Dataset Analysis


This repository contains the code and resources for analyzing a subset of the Cosmology and Astrophysics with MachinE Learning Simulations (CAMELS) Multifield Dataset (CMD). The goal of this project is to leverage deep learning techniques to determine fundamental cosmological parameters from simulated astrophysical datasets. The final report, included in this repository, documents the methodology, experiments, and results.

## Overview

The CAMELS Multifield Dataset is a vast simulated dataset designed for cosmology and astrophysics research. This project focuses on a specific subset of the dataset, using machine learning to predict cosmological parameters such as the matter density parameter ($\Omega_m$) and the power spectrum normalization ($\sigma_8$) from 2D simulation maps.

**Key Objectives:**
1. Train machine learning models to predict cosmological parameters from individual 2D maps.
2. Explore the effectiveness of combining multiple maps for enhanced predictions.
3. Experiment with transferring learning across different astrophysical fields.

## Project Structure

```plaintext
.
├── src/
│   ├── ModelTraining.py        # Main training script for the models
│   ├── MultipleMaps.py         # Code for handling and visualizing multiple maps
│   ├── TorchCheckpoints.pt     # PyTorch model checkpoint
│   ├── CAMELS_LOADER.py        # Custom dataset loader for CAMELS data
│   ├── HyperTuneTorch.py       # Script for hyperparameter tuning using PyTorch
│   ├── Main.ipynb              # Main Jupyter notebook for the project
│   ├── HyperTuneTF.py          # Hyperparameter tuning using TensorFlow (note TF was abandoned early in this project so script is not very funcitonal)
├── Report/
│   ├── ML_miniproject.pdf      # Report detailing the methodology, results, and analysis
│   ├── ML_miniproj.tex         # Latex source file for report
│   ├── ML_miniproj.synctex.gz
│   ├── UCL.png
├── HyperTuing/...              # Pickle files used for saving data during runs
└── README.md                   # This file
```

## Dataset Description

The subset used in this project comes from the IllustrisTNG simulation, focusing on 2D maps where a single astrophysical parameter is varied. The following cosmological and astrophysical parameters are considered:

| Variable      | Description |
|---------------|-------------|
| $\Omega_m$    | Matter density in the universe |
| $\sigma_8$    | Smoothness of the matter distribution |
| $A_{SN1}$, $A_{SN2}$ | Parameters controlling supernova feedback |
| $A_{AGN1}$, $A_{AGN2}$ | Parameters controlling black-hole feedback |

The dataset includes several astrophysical fields:

| Field                     | Prefix |
|---------------------------|--------|
| Gas density               | Mgas   |
| Gas velocity              | Vgas   |
| Gas temperature           | T      |
| Gas metallicity           | Z      |
| Neutral hydrogen density  | HI     |
| Electron number density   | ne     |
| Magnetic fields           | B      |
| Magnesium over Iron       | MgFe   |
| Dark matter density       | Mcdm   |
| Dark matter velocity      | Vcdm   |
| Stellar mass density      | Mstar  |
| Total matter density      | Mtot   |

## Methodology

1. **Data Preparation**:
    - Loaded and normalized the dataset using a custom `CAMELS_Dataset` class.
    - Applied data augmentation techniques but found no significant improvement in accuracy.

2. **Model Selection and Training**:
    - Developed a deep learning model using PyTorch for image-to-value regression.
    - Conducted extensive hyperparameter tuning to optimize model performance.
    - Evaluated the models based on both validation loss and a custom cost function to balance model size and accuracy.
  
   
    <img alt="HyperParameterTuning" src="https://github.com/user-attachments/assets/f3d8fe4c-1b54-48e4-8708-a8df95550b0c" width=700px></img>
    <img alt="ModelArchitecture" src="https://github.com/user-attachments/assets/38f92a0d-6a01-4248-8a89-079ff6eb961b" width=700px></img>


3. **Multi-Map Training**:
    - Experimented with training models on combinations of multiple maps, finding that including additional maps as channels yielded better results.
  
    <img alt="ExampleTrainingRun" src="https://github.com/user-attachments/assets/1c9079d3-3741-4f51-b7ad-e1c3f9c7d21e" width=700px></img>
    <img alt="MultipleMapsComparison" src="https://github.com/user-attachments/assets/ccbb0ebb-3cbb-44ae-8e5d-c86820fc6728" width=700px></img>


## Report Summary




The final report [ML_miniproj.pdf](Report/ML_miniproject.pdf) provides a comprehensive overview of the project. It includes:

- **Introduction**: An overview of the problem and objectives.
- **Data**: Description and preprocessing steps for the CAMELS dataset.
- **Methodology**: Detailed explanation of the machine learning models used, including the choice of architecture and training process.
- **Results**: Presentation of the results, including model performance metrics.
- **Conclusion**: Summary of findings and potential future work.

### Feedback on Report

- **Strengths**:
  - Professionally formatted with clear figures, captions, and tables.
  - Justified model choice with a well-described training process.
  - Clear presentation of results.
  
- **Areas for Improvement**:
  - Stronger connection with the underlying physics and parameters could enhance the report.
  - Inclusion of visualizations of the input data maps is recommended.

### Feedback on Code

- **Strengths**:
  - Reasonably well-formatted and easy-to-navigate notebooks.
  - Clear code with a good use of machine learning techniques.

- **Areas for Improvement**:
  - More markdown or commenting to clarify the purpose of each code block would be beneficial.

## Results

The project demonstrates the effectiveness of using deep learning to predict cosmological parameters from simulated astrophysical data. The analysis also highlights which map types are most predictive of these parameters, providing insights into the underlying physical processes.

## Potential Extensions

- **Transfer Learning**: Train models on one field and test them on another to evaluate cross-field predictive power.
- **Generative Modeling**: Develop generative models that can predict one field from another.

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## Running the Project

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/camels-multifield-dataset.git
    ```
2. Navigate to the `src` directory:
    ```bash
    cd camels-multifield-dataset/src
    ```
3. Open and run the `Main.ipynb` notebook in Jupyter to see the entire workflow, from data loading to model evaluation.

## References

- [CAMELS Multifield Dataset on arXiv](https://arxiv.org/abs/2109.10915)
-   Readthedocs.io. (2020). CAMELS — CAMELS 0.1 documentation. [Online]. Available at: [https://camels.readthedocs.io/en/latest/](https://camels.readthedocs.io/en/latest/) [Accessed 8 Mar. 2024].
-   Parameter Estimation. (2016). CAMELS - Parameter estimation. [Online]. Available at: [https://www.camel-simulations.org/parameter-estimation](https://www.camel-simulations.org/parameter-estimation) [Accessed 8 Mar. 2024].
-   Villaescusa-Navarro, F., Genel, S., Angles-Alcazar, D., Thiele, L., Dave, R., Narayanan, D., Nicola, A., Li, Y., Villanueva-Domingo, P., Wandelt, B.D., Spergel, D.N., Somerville, R.S., Zorrilla, M., Mohammad, F.G., Hassan, S., Shao, H., Wadekar, D., Eickenberg, M., Kaze, C., & Contardo, G. (2022). The CAMELS Multifield Data Set: Learning the Universe’s Fundamental Parameters with Artificial Intelligence. *The Astrophysical Journal Supplement Series*, 259(2), 61–61. doi:[https://doi.org/10.3847/1538-4365/ac5ab0](https://doi.org/10.3847/1538-4365/ac5ab0).
-   Readthedocs.io. (2021). Description — CAMELS Multifield Dataset 1.0 documentation. [Online]. Available at: [https://camels-multifield-dataset.readthedocs.io/en/latest/data.html#d-maps](https://camels-multifield-dataset.readthedocs.io/en/latest/data.html#d-maps) [Accessed 8 Mar. 2024].
-   scikit-learn. (2024). `sklearn.model_selection.ParameterGrid`. [Online]. Available at: [https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterGrid.html](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterGrid.html) [Accessed 8 Mar. 2024].
-   Dino, L. (2022). K-fold CV — Hyper-parameter tuning in Python. *Little Dino - Medium*. [Online]. Available at: [https://medium.com/@24littledino/k-fold-cv-hyper-parameter-tuning-in-python-4ad95880e477](https://medium.com/@24littledino/k-fold-cv-hyper-parameter-tuning-in-python-4ad95880e477) [Accessed 16 Mar. 2024].
-   lutzroeder. (2024). GitHub - lutzroeder/netron: Visualizer for neural network, deep learning and machine learning models. [Online]. Available at: [https://github.com/lutzroeder/netron](https://github.com/lutzroeder/netron) [Accessed 18 Mar. 2024].

## License

This project is licensed under the MIT License.

---

