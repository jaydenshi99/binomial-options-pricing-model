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
5. Run the Streamlit app: `streamlit run src/app.py`

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
