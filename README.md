# DeepML Quest Generator

AI-powered learning quest generator for machine learning problems. Generates interactive learning paths with mathematical explanations, exercises, and test cases.

## Features

- **Quest Generation** - Create 5-step learning paths for ML problems
- **Mathematical Content** - Detailed formulas, definitions, and theorems
- **Practice Exercises** - Auto-generated exercises with test cases
- **Multiple AI Backends** - OpenAI API or GitHub Copilot
- **Batch Processing** - Generate quests by category or all at once
- **Failure Recovery** - Track and retry failed generations

## Installation

```bash
# Clone the repository
git clone git@github.com:NeuronLab-ID/deepml-scraper.git
cd deepml-scraper

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env with your API keys
```

## Environment Variables

Create a `.env` file with:

```env
# Required for quest generation
OPENAI_API_KEY=your-openai-api-key

# Optional
GITHUB_TOKEN=your-github-token
PERPLEXITY_API_KEY=your-perplexity-api-key
```

## Usage

### Generate Quest for Single Problem

```bash
python quest_generator.py --problem-id 1
```

### Generate Quests by Category

```bash
python quest_generator.py --category "Linear Algebra"
python quest_generator.py --category "Machine Learning"
python quest_generator.py --category "Deep Learning"
```

### Generate All Missing Quests

```bash
python quest_generator.py --all
```

### Retry Failed Quests

```bash
# List failed quests
python quest_generator.py --list-failed

# Retry all failed quests
python quest_generator.py --retry-failed
```

### Use Different AI Backend

```bash
# OpenAI (default)
python quest_generator.py --problem-id 1 --backend openai

# GitHub Models (uses Copilot quota)
python quest_generator.py --problem-id 1 --backend github
```

## Project Structure

```
deepml-scraper/
├── quest_generator.py    # Main quest generation script
├── deepml_cli.py         # Problem scraper CLI
├── learning_cli.py       # Interactive learning CLI
├── problems/             # 270 ML problem definitions
│   └── problem_XXXX.json
├── quests/              # Generated learning quests
│   └── quest_XXXX.json
├── requirements.txt
├── .env.example
└── .gitignore
```

## Quest Structure

Each generated quest contains:

```json
{
  "problem_id": 1,
  "title": "Matrix-Vector Dot Product",
  "sub_quests": [
    {
      "step": 1,
      "title": "Vector Basics",
      "relation_to_problem": "Understanding vectors is required...",
      "math_content": {
        "definition": "A vector is...",
        "theorem": "..."
      },
      "key_formulas": [
        {"name": "Dot Product", "latex": "$\\mathbf{u} \\cdot \\mathbf{v}$", "description": "..."}
      ],
      "exercise": {
        "description": "Implement...",
        "starter_code": "def solution():\n    pass",
        "test_cases": [...]
      },
      "hint": "Consider using numpy...",
      "common_mistakes": ["Forgetting to..."]
    }
  ]
}
```

## Categories

Problems are organized into these categories:

- Linear Algebra (matrices, vectors, eigenvalues)
- Machine Learning (regression, classification, clustering)
- Deep Learning (neural networks, CNNs, RNNs)
- NLP (tokenization, embeddings, transformers)
- Computer Vision (image processing, object detection)
- Data Preprocessing (normalization, feature engineering)
- Probability (distributions, Bayesian methods)
- MLOps (deployment, monitoring)
- Optimization (gradient descent, convex optimization)

## Related Projects

- [NeuronLab Backend](https://github.com/NeuronLab-ID/backend) - FastAPI backend
- [NeuronLab Frontend](https://github.com/NeuronLab-ID/frontend) - Next.js web app

## License

MIT License
