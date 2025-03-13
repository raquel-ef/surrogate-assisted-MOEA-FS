# Surrogate-assisted and Filter-based Multiobjective Evolutionary Feature Selection for Deep Learning

## Overview
Feature selection (FS) for deep learning prediction models is a complex challenge. Traditional FS methods include:
- **Embedded methods**: Modify neural network weights to adjust attribute importance.
- **Filter methods**: Independent of learning algorithms but can limit prediction accuracy.
- **Wrapper methods**: Computationally expensive for deep learning.

This project proposes novel FS methods incorporating:
- **Multiobjective and many-objective evolutionary algorithms** for search strategies.
- **Surrogate-assisted approach** to reduce wrapper-type computational cost.
- **Filter-type objective functions** based on correlation and an adaptation of the reliefF algorithm.

The techniques were applied to air quality forecasting in Spain and indoor temperature prediction in a domotic house, yielding promising results.

---

## Project Structure
```plaintext
📂 config/                      # Configuration file
📂 data/                        # Dataset storage
📂 models/                      # Trained models
📂 notebooks/                   # Jupyter Notebooks with examples
 ├── Algorithm search.ipynb             # Best MOEA search
 ├── Comparison FS methods.ipynb        # Comparisons of FS methods
 ├── Problem search.ipynb               # Best problem search
📂 problems/                    # Feature selection problems definitions
 ├── FS_O1O2_LSTM.py
 ├── FS_O1O2O3_LSTM.py
 ├── FS_O1O2O3O4_LSTM.py
 ├── FS_O1O2O4_LSTM.py
 ├── FS_O3O4O2_LSTM.py
 ├── LR_wrapper.py
 ├── RF_wrapper.py
📂 src/                         # Source code
 ├── data_processing.py             # Preprocessing and cleaning
 ├── evaluation.py                  # Model evaluation scripts
 ├── utils.py                       # Utility functions
📂 Variables/               # Variable storage
📜 build_LSTM.py            # LSTM surrogate model construction
📜 requirements.txt         # Dependencies
```

## Installation
Ensure you have Python installed and install the following dependencies:
```sh
pip install -r requirements.txt
```


## Usage
1. **Prepare the dataset**. Place data in `.arff` format in the `data/` directory. Note that the data must be previously transformed using a sliding window method (see function `lags` in `utils.py` for this transformation).
2. **Train the surrogate model**. Train the surrogate model using `build_LSTM.py` file. The parameters should be previously configured in `config.py`.
3. **Search the best problem**. Run the notebook `Problem search.ipynb` to find the best problem (objectives combination).
4. **Search the best algorithm**. Once the best problem is selected run the notebook `Algorithm search.ipynb` to find the best algorithm. Note that by default the problem will be `FS_O1O2O3_LSTM.py`
5. **Compare with other feature selection methods**. Run the notebook `Comparisons FS methods.ipynb` to compare with other FS methods:

    - Wrapper multi-objective evolutionary FS method based on Linear Regression
    - Wrapper multi-objective evolutionary FS method based on Random Forest
    - Hybrid filter-wrapper FS method based on correlation and LSTM with deterministic search
    - Hybrid filter-wrapper FS method based on reliefF and LSTM with deterministic search
    - [CancelOut layer](https://link.springer.com/chapter/10.1007/978-3-030-30484-3_6)
    - Recursive feature elimination with cross-validation (RFECV) with Random Forest

## Citation
If you use this software in your work, please include the following citation:
```
@article{espinosa2023,
  title={Surrogate-assisted and filter-based multiobjective evolutionary feature selection for deep learning},
  author={Espinosa, Raquel and Jim{\'e}nez, Fernando and Palma, Jos{\'e}},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2023},
  publisher={IEEE}
}
```

## License
MIT License

