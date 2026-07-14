# NASA C-MAPSS Dataset Documentation

## 1. Dataset Overview
Data sets consists of multiple multivariate time series. Each data set is further divided into training and test subsets. Each time series is from a different engine i.e., the data can be considered to be from a fleet of engines of the same type. Each engine starts with different degrees of initial wear and manufacturing variation which is unknown to the user. This wear and variation is considered normal, i.e., it is not considered a fault condition. There are three operational settings that have a substantial effect on engine performance. These settings are also included in the data. The data is contaminated with sensor noise.

## 2. Project Objective

The task is to predict the number of remaining operational cycles of an unspecified system before failure in the test set, i.e.,the number of operational cycles still running after the last cycle.

## 3. Dataset Subsets

| Subset | Operating Conditions | Fault Modes |
|--------|----------------------|-------------|
| FD001  | ONE (Sea Level)      |  ONE (HPC Degradation)           |
| FD002  | SIX                  |         ONE (HPC Degradation)    |
| FD003  | ONE (Sea Level)      |          TWO (HPC Degradation, Fan Degradation)   |
| FD004  | SIX                  |  TWO (HPC Degradation, Fan Degradation)           |

## 4. Files

Training data: engine history continues until failure.

Testing data: engine history stops before failure.

RUL files: tell us how many cycles remain running after the test data stops.

## 5. Data Structure

Each row represents one engine at one specific operating cycle.

The column groups:
- Engine ID
- Cycle
- Operational settings
- Sensor measurements

## 6. Prediction Target

The RUL represents the number of additional cycle an engine can operate before failure after its test data stops.

## 7. Raw Data Policy

Original files are preserved in their original form and are not manually modified.

## 8. Dataset Source

NASA's Prognostics Center of Excellence repository.