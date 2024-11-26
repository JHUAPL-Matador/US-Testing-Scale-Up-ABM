# **Agent Based Model of the COVID-19 Testing and Diagnostics in the United States**

This repository contains the agent-based model (ABM) that was used to simulate testing and diagnostic of COVID-19 for the U.S. This model code was referenced in "U.S. National SARS-CoV-2 Test Scale-Up: Number of Tests Produced and Used Over Time and Impact on the COVID-19 Pandemic" that was published in *Lancet Public Health*. This repository includes the complete implementation of the model, all parameter settings, and scripts for running the experiments and analyzing results. The study exclusively uses open-source data, all of which is included in the repository. To ensure full reproducibility, we provide example model runs and detailed instructions for replicating the simulations and analyses presented in this work.

This work was funded for by the **Administration for Strategic Preparedness and Response (ASPR)** and the **Centers for Disease Control and Prevention (CDC)**. 

## Table of Contents
- [**Agent Based Model of the COVID-19 Testing and Diagnostics in the United States**](#agent-based-model-of-the-covid-19-testing-and-diagnostics-in-the-united-states)
  - [Table of Contents](#table-of-contents)
  - [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Data](#data)
    - [Sources](#sources)
  - [Visualizing the Results](#visualizing-the-results)
  - [License](#license)


## Getting Started
The following repo contains the scripts and data that were used to run the main results of the article. The notebook, titled `run-models.ipynb` initializes the runs. To start the simulation, the User, just need to run that notebook and install the prerequisite packages. We recommend uisng a HPC to run the simulations in parallel for Users that want to increase their number of simulations. 

The main scripts for each scenario with customized adjustments are located in the `model` folder, which includes four scripts:

- `msimC0.py` (C0)
- `msimC1.py` (C1)
- `msimC2.py` (C2)
- `msimC3.py` (C3)

## Prerequisites
The ABM is based on the Covasim 3.1.4 workflow developed by the Institute of Disease Modeling (IDM). 

The following packages are necessary for running the ABM:

- covasim 3.1.4 (2022-10-22)
- scipy 1.11.3
- pandas 1.5.3
- numpy 1.26.2
- joblib 1.3.2
- sciris 3.1.2


## Data 
The `data` folder contain the model input data for the COVASIM. This includes four data input files for scenarios C0 to C1:
- `usa-data-1.0-1.0.csv` (C0)
- `usa-data-1.0-0.2.csv` (C1)
- `usa-data-1.0-delay.csv` (C2)
- `usa-data-1.0-1.8.csv` (C3)

Additionally, this folder also includes the realtive transimissibility parameters (`param.soc.csv`) and the final variant-specific parameters (`param.var.csv`).  


### Sources:
- **COVID-19 Trends and Impact Surveys ([CTIS](https://cmu-delphi.github.io/delphi-epidata/api/covidcast-signals/fb-survey.html))**: Survey collected through Facebook about testing habits. We utilized this data up to June 2022.
- **COVID-19 Electronic Lab Reporting (CELR)**: Standardized lab reporting collected through the CDC Electronic Laboratory Reporting (ELR). This public health data is the transmission of digital laboratory reports, often from laboratories to state and local public health departments, healthcare systems, and CDC.


## Visualizing the Results
In order to reduce storage, the `results` folder includes a sample of 10 runs that was used in the manuscript. ABM output are automatically stored in the `results` folder.  Please feel free to reach out to the corresponding author or repo manager for access the full dataset.  


## License
[MIT License](doc:LICENSE.md)

*Copyright (c) 2024 The Johns Hopkins University Applied Physics Laboratory*

