# Integrated Agronomic Analytics System 🌾
**Machine Learning Yield Prediction & Computer Vision Plant Phenotyping**

## 📖 Executive Summary
Wheat contributes to ~20% of global calories and over A$10 billion annually to the Australian economy, yet yield outcomes are highly sensitive to complex, non-linear interactions between climate, cultivars, and management practices. 

This Master of Data Science capstone project bridges the gap between regional environmental data and micro-level plant traits. I developed a dual-pipeline data science system featuring a **Machine Learning yield forecasting model** and a **Computer Vision organ segmentation tool**, wrapped into an interactive dashboard to drive actionable agricultural decision-making.

## 🛠️ Technical Stack
* **Languages:** Python
* **Computer Vision:** PyTorch, Torchvision, Segmentation Models PyTorch (SMP)
* **Machine Learning:** Scikit-Learn (Random Forest, Gradient Boosting, SVR, PCA)
* **Data Engineering & Analysis:** Pandas, NumPy, Matplotlib, Seaborn
* **Deployment & Cloud:** HPC (Rangpur) for privacy-compliant model training

---

## 🧠 Pipeline 1: Regional Yield Prediction (Machine Learning)
Traditional simulation models often fail to capture complex environmental relationships or require immense computational overhead. This pipeline scales predictive analytics to regional levels.

* **Objective:** Forecast wheat yield (kg/plot) based on tabular environmental data, daily weather records, and cultivar types.
* **Models Evaluated:** Random Forest (RF), Support Vector Regression (SVR), Gradient Boosting (GB), and Principal Component Analysis (PCA) with Linear Regression.
* **Key Achievements:** * Handled severe data imbalances and high-dimensionality weather data through robust preprocessing and feature engineering.
  * Identified **Random Forest** as the optimal baseline model, providing highly accurate predictions while maintaining critical feature interpretability for stakeholders.

## 👁️ Pipeline 2: Fine-Grained Plant Phenotyping (Computer Vision)
To understand yield at the biological level, we need to extract traits (like head-leaf ratio or stem density) directly from field imagery.

* **Objective:** Perform precise semantic segmentation on field images to classify pixels into four categories: Background, Wheat Head, Stem, and Leaf.
* **Architectures Built:** U-Net, SegFormer, and DeepLabV3+.
* **Key Achievements:**
  * Successfully trained deep learning models to identify intricate plant structures.
  * Fine-tuned **DeepLabV3+**, which delivered the highest accuracy and proved especially robust at identifying heavily imbalanced minority classes (e.g., wheat stems).

### Visual Results: Image Segmentation
The images below demonstrate the DeepLabV3+ model's ability to take raw field images, compare them to ground truth annotations, and generate accurate predicted masks.

![Segmentation Example 1](assets/segmentation_example_0.png)
![Segmentation Example 2](assets/segmentation_example_1.png)
![Segmentation Example 3](assets/segmentation_example_2.png)

---

## 💡 Business Impact & Future Scope
* **Decision Dashboard:** Synthesized the outputs into a decision-making dashboard, allowing non-technical agricultural stakeholders to analyze crop data easily.
* **Scalability:** The system is designed to eventually integrate CV-derived phenotypic traits directly into the ML yield models for a fully unified, multi-modal forecasting engine.
* **Data Governance:** Ensured full compliance with institutional and open-data agreements (NVT/SILO), utilizing localized and HPC processing to guarantee data privacy.

---
*Project completed as part of the Master of Data Science program at The University of Queensland.*
