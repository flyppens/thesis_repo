# thesis_repo

This is the repository for my thesis. The basis of the code for the transformer is taken from: https://github.com/hkproj/pytorch-transformer. I edited some of the code to work for my purposes and changed the parameters and datasets which were used. Models were trained via GCP on an NVIDIA L4 with 4vCPU and 32GB RAM and 40GB disk space. 

Below is a more detailed explanation of the repository structure. 

## LLMCarbon
Code used to obtain carbon output predictions for LLMCarbon:
- model.py
- llmcarbon_tutorial.py
- embodied.py -> for embodied carbon predictions

Data used by LLMCarbon:
- database.csv
- hardware.csv

Output of LLMCarbon:
- sensitivity_row11.csv

Code used for accuracy and sensitivity analyses:
- Sobol_analysis.py
- feat_importance_regr.py
- regr_new.py

## OpenCarbonEval
Code used to obtain carbon output predictions for OpenCarbonEval:
- code_predictions
   - opencarboneval_main.ipynb

Data used by OpenCarbonEval:
- data
  - OpenCarbonEval_dataset.csv
  - model_emission.csv
  - notable_ai_models.csv
  - valid_models.csv

Code used for accuracy and sensitivity analyses:
- Sobol_analysis.py
- lin_regr.py
- sens_pfi_.py
- sensitivity_pfi_gpt_row11.csv

## Transformer
The model:
- model.py

Code to start training using CodeCarbon and Weights & Biases tracking:
- train.py
- train_wb_japan.py

Configuration of the models:
- config.py
- config_japan.py

Code used to call the data used to train the model:
- dataset.py

## Executive Summary
- For a summary about the research, how it was set up, what the results were, what we learned, and what could be improved upon, see:
- exec_summary_thesis.pdf

