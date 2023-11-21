# Machine Learning versus Physicians

This repository contains the code and resources for our project titled: "Machine-learning based glucose predition". The project uses continuous glucose and physical activity data to predict glucose levels in patients with type 2 diabetes mellitus. The project is part of the [Maastricht Study](https://www.demaastrichtstudie.nl/), a large-scale, prospective, observational cohort study that focuses on the etiology, pathophysiology, complications and comorbidities of type 2 diabetes mellitus.

This work is based on the paper:

"Machine learning-based glucose prediction with use of continuous glucose and physical activity monitoring data: The Maastricht Study"  
Published in PLoS One. 2021; 16(1): e0245157.  
[DOI: 10.1371/journal.pone.0253125](https://doi.org/10.1371/journal.pone.0253125)

## Repository Structure

**algorithms/**  
Contains various versions of the algorithm and the associated results. 

**data/**  
Holds all the data files used in the project, including laboratory data from the emergency department, clinical and baseline data from the ED sepsis study, and data from the questionnaires from the internists.

**docs/**  
Includes all the necessary documentation for this project, as well as documentation from external packages and/or software systems. 

**figures/**  
Stores all the figures produced by the different experiments conducted in our study.

**models/**  
Contains the specific files with pre-processed data that serve as "data models" (i.e., not as an algorithm, but e.g., the lab or lab + clinical datasets). _TODO: consider renaming this to 'datasets'_

**notebooks/**  
Holds all Jupyter notebooks used during the project. These were mainly involved during code design and simple experiments. 

**src/**  
Contains all the source code from the project.