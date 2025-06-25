import argparse
import json
import os
from openai import OpenAI
import time
import numpy as np

client = OpenAI(api_key="XXX")


def parse_args():
    parser = argparse.ArgumentParser(description='Run VoiceBench evaluation')
    parser.add_argument('--input_path', type=str, default=None, help='Input JSONL file')
    parser.add_argument('--subset', type=str, default='commoneval', help='VoiceBench subset to evaluate on')
    return parser.parse_args()

"""
LLM_OPTIONS taken from environment variables

"""

meta_prompt_open = """
I need your help to evaluate the performance of several models in the speech interaction scenario. The models will receive a speech input from the user, which they need to understand and respond to with a speech output.
Your task is to rate the model's responses based on the provided user input transcription [Instruction] and the model's output transcription [Response].

Please evaluate the response on a scale of 1 to 5:
1 point: The response is largely irrelevant, incorrect, or fails to address the user's query. It may be off-topic or provide incorrect information.
2 points: The response is somewhat relevant but lacks accuracy or completeness. It may only partially answer the user's question or include extraneous information.
3 points: The response is relevant and mostly accurate, but it may lack conciseness or include unnecessary details that don't contribute to the main point.
4 points: The response is relevant, accurate, and concise, providing a clear answer to the user's question without unnecessary elaboration.
5 points: The response is exceptionally relevant, accurate, and to the point. It directly addresses the user's query in a highly effective and efficient manner, providing exactly the information needed.

Below are the transcription of user's instruction and models' response:
### [Instruction]: {prompt}
### [Response]: {response}

After evaluating, please output the score only without anything else.
You don't need to provide any explanations.
"""

def generate_text_chat(client, *args, **kwargs):
    e = ''
    for _ in range(25):
        try:
            response = client.chat.completions.create(*args, **kwargs)
            time.sleep(0.5)
            if response is None:
                time.sleep(30)
                continue
            return response
        except Exception as e:
            time.sleep(30)
    return None


def generate_scores(item):

    prompt = meta_prompt_open.replace("{prompt}", item['question']).replace('{response}', item['model_answer'])
    rtn = [
        item.message.content.strip() for item in generate_text_chat(
            client=client,
            model='gpt-4o-mini',
            messages=[{"role": "system",
                       "content": "You are a helpful assistant who tries to help answer the user's question."},
                      {"role": "user", "content": prompt}],
            max_tokens=1024,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            temperature=0.5, top_p=0.95, n=3
        ).choices
    ]
    return rtn


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

def run_eval_alpaca(input_path, output_path_md, output_path_json, subset='alpacaeval_full'):
    model_answers = []
    questions = []
    trajectories = []
    scores_full = []
    scores_mean = []
    scores_std = []
    ids = []
    total_count = 0
    web_search_count = {0:0, 1:0, 2:0, 3:0}
    turns_count = {0:0, 1:0, 2:0, 3:0}

    with open(input_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            ids.append(data['id'])
            model_answers.append(data['model_answer'].strip().lower())
            questions.append(data['question'])
            trajectories.append(data['trajectory'])

    for i in range(len(model_answers)):

        scores = generate_scores({'question':questions[i], 'model_answer':model_answers[i]})

        valid_scores = [int(score) for score in scores if score in ['1','2','3','4','5']]
        scores_full.append(valid_scores)
        scores_mean.append(np.mean(valid_scores))
        scores_std.append(np.std(valid_scores))


        turns = len(trajectories[i])
        local_web_search_count = 0
        for turn in trajectories[i]:
            if turn['action']['type'] == 'web_search':
                local_web_search_count += 1
        web_search_count[local_web_search_count] += 1
        turns_count[turns] += 1
        total_count += 1
        with open(output_path_json, 'a') as f:
            f.write(json.dumps({'id':ids[i], 'scores':valid_scores}) + '\n')

    with open(output_path_md, 'w') as f:
        # Write header and summary
        f.write(f"# VoiceBench - {subset} Evaluation Report\n\n")
        f.write("## Summary Statistics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Total Samples | {total_count} |\n")
        f.write(f"| Average Score | {np.mean(scores_mean):.2f} |\n")
        f.write(f"| Average Std | {np.mean(scores_std):.2f} |\n")

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

    print(f"\nResults saved to {output_path_md} and {output_path_json}")

def main():
    args = parse_args()
    input_dir = "/".join(args.input_path.split('/')[:-1])
    input_file = args.input_path.split('/')[-1].split('.')[0]
    output_path_md = f"{input_dir}/{input_file}_eval.md"
    output_path_json = f"{input_dir}/{input_file}_eval.jsonl"
    if args.subset in ['openbookqa']:
        run_eval_mcq(args.input_path,output_path_md,args.subset)
    elif args.subset in ['alpacaeval', 'alpacaeval_full','commoneval']:
        run_eval_alpaca(args.input_path,output_path_md,output_path_json,args.subset)
    

if __name__ == "__main__":
    main()
    # controller = Controller(operation_mode=Mode.QA_EVAL, io_mode=Mode.TEXT_2_TEXT_CASCADED, max_iterations=3)
    # output, transcript = controller.qa_eval({'instruction':'Who is the current president of South Korea?'})
    # print(output)
    # print(transcript)