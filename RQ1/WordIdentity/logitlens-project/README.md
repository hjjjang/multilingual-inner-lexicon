# LogitLens Project

## Overview
The LogitLens project is designed to provide functionality for analyzing and interpreting the behavior of language models in distinguishing between words and non-words. It includes a `LogitLens` class that extends the capabilities of the `WordNonwordClassifier`, allowing users to set up tokenizers and run logit lens analyses on input data.

## Installation
To set up the project, clone the repository and install the required dependencies. You can do this by running the following commands:

```bash
git clone <repository-url>
cd logitlens-project
pip install -r requirements.txt
```

## Requirements
The project requires the following Python packages:

- torch
- transformers
- pandas
- tqdm

Make sure you have these packages installed in your Python environment.

## Usage
To use the `LogitLens` class, you can import it from the `logitlens` module in your Python scripts. Here is a basic example of how to set up and run the logit lens functionality:

```python
from src.logitlens import LogitLens

# Initialize the LogitLens object
logit_lens = LogitLens()

# Set up the tokenizer
logit_lens.setup_tokenizer()

# Run logit lens analysis on your data
results = logit_lens.run_logit_lens(df, distance_metric='cosine', k=5, type='simple_split')
```

## Contributing
Contributions to the LogitLens project are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.