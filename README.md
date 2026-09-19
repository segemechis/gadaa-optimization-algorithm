# Gadaa Optimization Algorithm (GOA)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the MATLAB source code and benchmark results for the **Gadaa Optimization Algorithm (GOA)**, a novel multi-population metaheuristic inspired by the Gadaa system — the indigenous democratic governance institution of the Oromo people of Ethiopia, inscribed on the UNESCO Representative List of Intangible Cultural Heritage of Humanity in 2016.

GOA organises 50 search agents into 5 parallel Gogessa clans across six hierarchical age-grade tiers, with each grade level implementing a distinct search operator calibrated to its position on the exploration-to-exploitation spectrum.


---

## Repository Structure

gadaa-optimization-algorithm/
│
├── Code/
│ ├── GOA_fixed.m # Main GOA algorithm (full implementation)
│ ├── GOA_grades.m # Grade structure and grade update operators
│ ├── GOA_butta_deb.m # Butta ceremony and Deb feasibility ordering
│ └── GOA_tracking.m # Convergence tracking and performance logging
│
└── Results/
├── CEC2017_Benchmark/ # CEC-2017 results at D=10, 30, 50
├── CEC2020_Benchmark/ # CEC-2020 results at D=5, 10, 15, 20
├── CEC2022_Benchmark/ # CEC-2022 results at D=10, 20
├── Moving_Peaks/ # Moving Peaks Benchmark (11 scenarios)
├── Medical_Segmentation/# Multi-level biomedical image thresholding
├── Ablation_Study/ # Nine-variant ablation study results
└── Sensitivity_Analysis/# Parameter sensitivity analysis results


---

## Key Results

| Benchmark | Dimension | GOA Friedman Rank | χ² | p-value |
|-----------|-----------|-------------------|-----|---------|
| CEC-2017  | D=10,30,50 | **2.62** (best) | 222.71 | <0.001 |
| CEC-2020  | D=5,10,15,20 | **2.53** (best) | 86.87 | <0.001 |
| CEC-2022  | D=10,20 | **2.96** (best) | 93.97 | <0.001 |
| Moving Peaks | 11 scenarios | **1.91** (best among standard algorithms) | — | — |
| Medical Thresholding | 5 images × 4 levels | **2.55** (best) | — | — |

GOA was evaluated against 12 state-of-the-art competitors: PSO, GWO, WOA, DE, RIME, MVO, TLBO, SO, SHO, SCSO, SLO, and CCO.

---

## Requirements

- MATLAB R2021b or later
- CEC benchmark function evaluation code (available from the official CEC competition organisers)

---

## Usage

```matlab
% Run GOA on a benchmark function
% Load your benchmark function handle first, then:
[best_val, best_pos, curve] = GOA_fixed(fobj, lb, ub, dim, N, T_max);
```

---

## Citation

If you use this code or results in your research, please cite:

Sena Gemechis File, "Gadaa Optimization Algorithm: A Governance-Inspired
Multi-Population Metaheuristic for Global Optimization,"
Scientific Reports, 2026 (under review).


---

## Author

**Sena Gemechis File**  
Faculty of Electrical and Computer Engineering  
Jimma Institute of Technology, Jimma University, Ethiopia  
ORCID: [0009-0007-9742-9530](https://orcid.org/0009-0007-9742-9530)  
Email: senagemechis1994@gmail.com

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
