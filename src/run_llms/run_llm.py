import argparse
from gpt_runner import GPTRunner
from gemini_runner import GeminiRunner
from claude_runner import ClaudeRunner

model_ids = {
    'llama': "meta-llama/Llama-3.1-8B-Instruct",
    'deepseek': "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    'llama_uncensored': "Orenguteng/Llama-3.1-8B-Lexi-Uncensored",
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run LLM on Excel data')
    parser.add_argument('--model_id', type=str, required=True, help='Model ID: llama, llama_uncensored, gpt, deepseek, gemini')
    parser.add_argument('--prompst_path', type=str, required=True, help='Path to the prompts csv file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output CSV file')
    parser.add_argument('--save_every', type=int, default=10, help='Save progress every N rows')

    args = parser.parse_args()

    if args.model_id == 'gpt':
        runner = GPTRunner(args.save_every, model_id = 'gpt')
    elif args.model_id == 'gemini':
        runner = GeminiRunner(args.save_every, model_id = "gemini-2.5-flash")
    elif args.model_id == 'claude':
        runner = ClaudeRunner(args.save_every, model_id = "claude-sonnet-4-20250514")
    else:
        raise ValueError("Invalid model ID. Choose from: gpt, claude, gemini")
    
    df = runner.read_csv(args.excel_path, args.sheet_name)

    print('Read prompts')

    model_pipeline = runner.connect()
    runner.run_llm(model_pipeline, df, args.output_path)