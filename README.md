# A mathematical model of vagal-mediated insulin secretion

This repository contains the code for the simulation of insulin dynamics based on the paper "Vagal nitric oxide action represses pancreatic insulin release in male mice" (in preparation). The simulation implements a mathematical model that describes the relationship between glucose concentration, vagal activity and insulin concentration.

# Data requirements
The code requires two CSV files:

241010Estimated_parameter.csv: Contains model parameters (k1-k6)

241010m3m4_sti.csv: Contains time series data for glucose, insulin, and vagal nerve activity

# Visualization
For each condition and measurement, the code generates five plots:
Glucose concentration (Gc), Insulin levels (I), Vagal nerve activity (cVNA), Activation factor (A), and Repression factor (R).
