# Swarm Translation System

A robust, parallelized system for translating biblical texts across multiple
languages using DSPy and modern LLMs.

## Overview

This system provides a scalable framework for translating biblical texts into
various languages. It's designed to handle large-scale translation tasks
efficiently through parallel processing while maintaining translation quality
and consistency.

## Features

- **Parallel Processing**: Utilizes multiprocessing to handle multiple languages
  and texts simultaneously
- **Resumable Operations**: Can resume from interrupted translation runs
- **Flexible Input Formats**: Supports CSV and JSONL input files
- **Configurable Scenarios**: Language-specific translation scenarios with
  custom parameters
- **Progress Tracking**: Detailed logging and completion markers
- **Rate Limiting**: Built-in delays to respect API rate limits
- **Error Handling**: Robust error recovery and logging

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set up environment variables in `.env`:

```
OPENAI_API_KEY=your_api_key_here
```

3. Prepare your translation plan in `PLAN.csv` with the following columns:

- LANGUAGE_NAME
- CODE
- COUNTRY
- REGION
- SCENARIO_FILEPATH

## Usage

### Basic Usage

```bash
python run_all_scenarios.py
```

### Command Line Options

- `--resume`: Resume from last completed scenario
- `--test`: Use test texts instead of full input files
- `--limit N`: Process only N texts per scenario
- `--scenario CODE`: Process specific language code(s)
- `--sequential`: Process scenarios sequentially
- `--no-parallel-texts`: Disable parallel text processing within scenarios

### Example Commands

```bash
# Run all scenarios
python run_all_scenarios.py

# Test mode with limited texts
python run_all_scenarios.py --test --limit 5

# Resume interrupted run
python run_all_scenarios.py --resume

# Process specific languages
python run_all_scenarios.py --scenario es --scenario fr
```

## Output Structure

```
translation_results/
├── [language_code]/
│   ├── translations_[timestamp].jsonl
│   └── completed.txt
└── translation_run.log
```

## Translation Scenarios

Each language requires a scenario file that defines:

- Input file format and fields
- Translation parameters
- Language-specific instructions

Example scenario structure:

```json
{
  "name": "Spanish Translation",
  "input": {
    "file": "input/es_texts.csv",
    "format": "csv",
    "id_field": "verse_id",
    "content_field": "text"
  },
  "parameters": {
    "temperature": 0.7,
    "max_tokens": 1000
  }
}
```

## Logging

The system maintains detailed logs:

- Main log: `translation_run.log`
- Process-specific logs: `translation_run_[process_id].log`

## Error Handling

- Failed translations are logged but don't stop the process
- Each scenario tracks successful/failed translations
- System can resume from interruptions

## Performance Considerations

- Parallel processing is automatically adjusted based on:
  - Available CPU cores
  - Number of texts to process
  - API rate limits
- Small batches (< 10 texts) are processed sequentially
- Built-in delays prevent API rate limit issues

## Troubleshooting

### Daemonic Process Error

If you encounter the error "daemonic processes are not allowed to have
children", this is a limitation of Python's multiprocessing when running in
certain environments. To resolve this:

1. Run the script with the `--sequential` flag:

```bash
python run_all_scenarios.py --sequential
```

2. Or use the `--no-parallel-texts` flag to disable parallel text processing:

```bash
python run_all_scenarios.py --no-parallel-texts
```

3. If you need parallel processing, ensure you're running the script directly
   (not through another process) and not in a daemon context.

### Common Issues

- **Memory Usage**: For large datasets, consider using the `--limit` flag to
  process fewer texts at once
- **API Rate Limits**: The system includes built-in delays, but you may need to
  adjust them based on your API provider's limits
- **File Permissions**: Ensure the output directory is writable
- **Process Limits**: Some systems limit the number of processes. Use
  `--sequential` if you hit these limits

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

[Your License Here]
