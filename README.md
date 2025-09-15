# Binomial Options Pricing Model

An interactive web application for visualizing and understanding the binomial options pricing model.

## Features

- Interactive binomial tree visualization
- Real-time parameter adjustment
- Support for European and American options
- Greeks calculation and visualization
- Educational tutorials and explanations

## Quick Start

1. Clone the repository
2. Create a virtual environment: `python3 -m venv .venv`
3. Activate the environment: `source .venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`

### Running the Application

**Web Interface (Recommended):**
```bash
streamlit run src/app.py
```

**Command Line Interface:**
```bash
python src/cli.py --help
python src/cli.py --S0 100 --K 105 --T 0.25 --r 0.05 --sigma 0.2 --steps 10 --type call
```

**Python Library:**
```python
from src.binomial_model import BinomialModel
model = BinomialModel(S0=100, K=105, T=0.25, r=0.05, sigma=0.2, n_steps=10)
print(model)
```

## Project Structure

```
binomial-options-pricing-model/
├── src/                    # Source code
│   ├── binomial_model.py   # Core binomial model implementation
│   ├── visualization.py    # Visualization functions
│   ├── app.py             # Streamlit web application
│   └── utils.py           # Utility functions
├── notebooks/             # Jupyter notebooks for development
├── tests/                 # Test files
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Development

To run tests:
```bash
pytest tests/
```

To run the development server:
```bash
streamlit run src/app.py
```

## License

MIT License - see LICENSE file for details.
