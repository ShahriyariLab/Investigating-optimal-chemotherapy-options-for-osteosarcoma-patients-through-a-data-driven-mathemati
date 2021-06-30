# Investigating optimal chemotherapy options for osteosarcoma patients through a data-driven mathematical model

All tumors are unique, so they might respond differently to the same treatment. In this work, we develop a mathematical model of the interactions between key players in osteosarcoma microenvironment and the most common chemotherapy drugs in three clusters of tumors with distinct immune compositions. We then study the behaviors of cells and cytokines in the tumor microenvironment under the effects of chemotherapy, investigate the responses in each cluster to different treatment regimens and various treatment start times, as well as suggest optimal dosages for tumors of each cluster.

This repository contains the following scripts:
qspmodel.py: python classes and methods for Mathematical model of immune response in osteosarcoma
chemo_qspmodel.py: python classes and methods for Mathematical model of osteosarcoma microenvironment with chemotherapy drugs
MAP_dynamics.ipynb: python notebook for analysis and plotting of dynamics of osteosarcoma microenvironment with MAP treatment
sensitivity_analysis.ipynb: python notebook for sensitivity analysis of chemotherapy-related parameters
chemo_resistant_dynamics.ipynb: python notebook for analysis and plotting of dynamics of osteosarcoma microenvironment with MAP treatment when cancer cells are resistant to certain chemotherapy drugs
vary_start_treatment.ipynb: python notebook for analysis and plotting of dynamics of osteosarcoma microenvironment under MAP treatment with different treatment start times
different_treatment_regimens.ipynb: python notebook for analysis and plotting of dynamics of osteosarcoma microenvironment with different treatment regimens
optimal_dosage.ipynb: python notebook for optimization of MAP dosages and plotting of dynamics of osteosarcoma microenvironment with the optimal dosages

All data sets needed to run the code are in 'input' folder.

If using any parts of this code please cite:
Le, T; Su, S; Shahriyari, L. Investigating optimal chemotherapy options for osteosarcoma patients through a mathematical model. Submitted, 2021. 
