source .venv/bin/activate
OUTPUT_NAME="${1:-0}.json"
python grade/scoring.py --module src.agent.graph --output-json "outputs/${OUTPUT_NAME}"