# Readme: Decoding single-crystal lithium growth through SEI-Omics

## 0. Environment
We use the Python language, and all the Python/Jupyter files can work in the environment as we provided in the `requirements.txt`. You could use pip or conda to install the Python libraries like `shap` and `scikit-learn`. Some simple functions were supplemented in `utils.py`.

## 1. Dataset
All cryo-TEM dataset characterizations (overall 81 high-quality samples) were performed using a transmission electron microscope (TEM, FEI Talos-S) operating at 200 kV, equipped with a Gatan 698 cryo-transfer holder. All the files can be found in the path `Dataset_Process/Dataset`. They are the dataset obtained from the same experimental samples except for the difference in feature (characteristics) numbers.

- `final_data.xlsx` is used for the MOR model.
- `predict.xlsx` is used for training the GWD model and GDS model; the `predict.xlsx` just has a concrete GWD or GDS size label based on the `final_data.xlsx`, so they are in fact the same datasets.
- `original.xlsx` has more SEI characteristics including 'Li longitudinal growth size', 'Li longitudinal growth maximum size', 'The difference between the maximum longitudinal growth size and average size', 'λ value', and so on. We use this dataset to confirm the effectiveness of 'λ value' for representing our SEI system.
- `lambda_data.xlsx` is the version only adding 'λ value' based on the `predicted_data.xlsx`.

## 2. Pre-Process
Ahead of model training, we carried out some experiments to determine the number of our classification target in `Dataset_Process/GWD(GDS)_results_of_multi_classification.py`. To confirm the classification and label division (namely, determine a threshold) method is reasonable, we used some clustering methods like PCA and t-SNE and some data analysis skills in `Dataset_Process/clustering.ipynb`.

## 3. Model Training
The model training files can be found in the directory `Model_and_Results`: `MOR(GDS, GWD) model_training.ipynb`. Details can be found in code annotations and supplementary information of our paper. Our best models (MOR, GWD, GDS model) were saved in the directory `Models_and_Results`.

## 4. Interpretability Analysis
The interpretability analysis results were based on our trained best models. Results of Shap method for analyzing MOR/GWD/GDS model can be found in the directory files `Models_and_Results/Interpretability.ipynb`.

## 5. Detailed Analysis with Feature λ
To further analyze the importance of λ, RFECV experiments and some fitting methods to confirm the effectiveness of 'λ value' can be found in `Lambda_Analysis/lambda_importance.ipynb`, or GWD/GDS analysis.
