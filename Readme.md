# A CROP RECOMMENDATION SYSTEM FOR PLATEAU STATE

An AI-powered crop recommendation system for Plateau State, Nigeria. Built with machine learning, it takes soil and climate measurements as input and recommends the most suitable crop to grow — along with the best-matching Local Government Areas (LGAs) to cultivate in.


## PURPOSE

Smallholder farmers and agricultural planners in Plateau State often lack access to data-driven guidance on which crops suit their land. This tool bridges that gap by combining a trained classification model with local climate and soil knowledge, providing explainable, location-specific recommendations. Clock ittttttttttt


## FEATURES

- Crop recommendation from 7 soil/climate inputs (N, P, K, pH, Temperature, Rainfall, Salinity)
- Top 3 best-matching LGAs ranked with plain-language explanations
- Visual parameter comparison — see how your inputs compare to the ideal range
- Feature influence chart — understand which factors drove the recommendation
- Covers 9 crops across 17 Plateau State LGAs (Highland & Lowland zones)

---

## FILE STRUCTURE

```
plateau-crop-advisor/
│
├── interface.py            # Main Streamlit application
│
├── models/
│   ├── best_model.pkl      # Trained classification model
│   ├── scaler.pkl          # Feature scaler (StandardScaler)
│   └── label_map.pkl       # Crop label encoder + feature list
│
├── Training.ipynb          # The training work was done here
├── .gitignore
└── README.md
```

---

## Technologies Used

  Python 3.x ---- Core language 
  Streamlit----- Web interface 
  scikit-learn --- Model training and scaling 
  SHAP ------- Model explainability 
  NumPy and  Pandas ---- Data handling 
  Matplotlib ------ charts and visualisations 

---

## Getting Started

**1. Clone the repository**
```bash
git clone 
cd plateau-crop-advisor
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add model files**

Place your trained model files in the `models/` folder:
- `best_model.pkl`
- `scaler.pkl`
- `label_map.pkl`

**4. Run the app**
```bash
streamlit run interface.py
```

## Crops Covered

Irish Potato · Tomato · Groundnut · Pepper · Cabbage · Maize · Wheat · Pineapple · Onion


## Input Parameters
  N  kg/ha ------- Nitrogen content in soil 
  P  kg/ha ------- Phosphorus content in soil 
  K  kg/ha ------- Potassium content in soil 
  pH Soil -------- acidity/alkalinity 
  Temperature °C ----- Average ambient temperature 
  Rainfall mm ------ Annual or seasonal rainfall 
  Salinity dS/m ----- Soil electrical conductivity 


## Notes

- LGA matching uses a scoring system based on climate zone adjustments (Highland vs Lowland) and per-LGA soil bias profiles
- Model explainability is powered by SHAP TreeExplainer — the feature influence chart reflects the model's actual decision, not a generic importance score
- To update the model, retrain in your notebook, re-export the `.pkl` files, and replace the files in the `models/` folder
CIAO
