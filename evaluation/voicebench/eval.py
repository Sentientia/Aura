import argparse
import json
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Run VoiceBench evaluation')
    parser.add_argument('--input_path', type=str, default=None, help='Input JSONL file')
    parser.add_argument('--subset', type=str, default='openbookqa', help='VoiceBench subset to evaluate on')
    return parser.parse_args()

"""
LLM_OPTIONS taken from environment variables

"""

def run_eval_mcq(input_path, output_path, subset='openbookqa'):
    reference_answers = []
    model_answers = []
    trajectories = []
    correct_count = 0
    total_count = 0
    web_search_count = {0:0, 1:0, 2:0, 3:0}
    turns_count = {0:0, 1:0, 2:0, 3:0}

    with open(input_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            reference_answers.append(data['reference_answer'].strip().lower())
            model_answers.append(data['model_answer'].strip().lower())
            trajectories.append(data['trajectory'])

    for i in range(len(reference_answers)):
        if reference_answers[i] == model_answers[i]:
            correct_count += 1
        turns = len(trajectories[i])
        local_web_search_count = 0
        for turn in trajectories[i]:
            if turn['action']['type'] == 'web_search':
                local_web_search_count += 1
        web_search_count[local_web_search_count] += 1
        turns_count[turns] += 1
        total_count += 1

    with open(output_path, 'w') as f:
        # Write header and summary
        f.write(f"# VoiceBench - {subset} Evaluation Report\n\n")
        f.write("## Summary Statistics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Total Samples | {total_count} |\n")
        f.write(f"| Correct Answers | {correct_count} ({correct_count/total_count*100:.1f}%) |\n\n")

        # Write detailed web search distribution
        f.write("## Web Search Distribution\n\n")
        f.write("| Number of Searches | Count |\n")
        f.write("|-------------------|-------|\n")
        total_searches , total_counts = 0,0
        for searches, count in sorted(web_search_count.items()):
            f.write(f"| {searches} | {count} |\n")
            total_searches += searches * count
            total_counts += count
        f.write(f"| Average Web Searches | {total_searches/total_counts:.2f} |\n")
        f.write("\n")

        # Write detailed turns distribution
        f.write("## Turns Distribution\n\n")
        f.write("| Number of Turns | Count |\n")
        f.write("|----------------|-------|\n")
        total_turns , total_counts = 0,0
        for turns, count in sorted(turns_count.items()):
            f.write(f"| {turns} | {count} |\n")
            total_turns += turns * count
            total_counts += count
        f.write(f"| Average Turns | {total_turns/total_counts:.2f} |\n")

    print(f"\nResults saved to {output_path}")

def main():
    args = parse_args()
    input_dir = "/".join(args.input_path.split('/')[:-1])
    input_file = args.input_path.split('/')[-1].split('.')[0]
    output_path = f"{input_dir}/{input_file}_eval.md"
    if args.subset in ['openbookqa']:
        run_eval_mcq(args.input_path,output_path,args.subset)

if __name__ == "__main__":
    main()
    # controller = Controller(operation_mode=Mode.QA_EVAL, io_mode=Mode.TEXT_2_TEXT_CASCADED, max_iterations=3)
    # output, transcript = controller.qa_eval({'instruction':'Who is the current president of South Korea?'})
    # print(output)
    # print(transcript)