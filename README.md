# 🏌️‍♀️ Data-Driven Golf Strategy Optimization

This project investigates how **machine learning**, **Bayesian statistics**, and **spatial modelling** can be used to simulate and optimise golf strategy under uncertainty. Combining theoretical rigor with practical insight from collegiate-level play, it aims to create an intelligent decision-support system for golfers.

## 🔍 Goals

- Model golf shot dispersion and expected performance using **Gaussian Process Regression (GPR)**
- Integrate **Bayesian modelling** with **loss functions** to reflect risk preferences (e.g., aggressive vs conservative play)
- Use spatial data to visualise and evaluate course strategies
- Explore how club selection and aimpoint influence expected strokes to hole out (ESHO)

## 📌 Features

- Shot outcome simulation using custom or TrackMan-derived dispersion models
- 2D spatial hole representations built using QGIS and Python (Shapely, matplotlib)
- Risk-based strategy comparisons across club/aimpoint combinations
- Modular architecture supporting visualisation, modelling, and simulation layers

## 🧠 Background

This repository supports my Kenneth Cooke Summer Research Fellowship and will evolve into my senior thesis. It builds on golf analytics methods like **strokes gained** and **DECADE**, aiming to make them more flexible and personalised by:
- Building Bayesian priors from real-world data
- Modelling terrain and shot zones using polygonal course geometry
- Allowing strategy adjustments based on player tendencies and goals

## 🌱 Long-Term Vision

- Integrate reinforcement learning to simulate full-hole and full-round strategies
- Expand terrain realism with slope, lie type, and weather considerations
- Create a user facing app where players can upload data and receive personalised club/aimpoint guidance - a “**data caddie**”

## 🛠️ Setup

Clone the repository:

```bash
git clone https://github.com/fedegudom47/golfModeL47.git
cd golfModeL47
```

Install required packages:

```bash
pip install -r requirements.txt
```

## 📬 Contact

Developed by Federica Domecq with the advising of Professor Gabriel Chandler and Johanna Hardin from Pomona College. Made possible my Kenneth Cooke Summer Fellowship.