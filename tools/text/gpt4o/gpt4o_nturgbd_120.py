from openai import OpenAI
from tqdm import tqdm 
import os


my_api_key = "$THE_API_KEY$"
my_base_url = "$THE_BASE_URL$"
client = OpenAI(api_key=my_api_key, base_url=my_base_url)

def read_action_names(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        action_names = [line.strip() for line in lines if line.strip()]
    return action_names

dataset_name = "nturgbd_120"
action_names = read_action_names(f"label_map/{dataset_name}.txt")

dialogs_template = [
# Round 1
"Describe the movement process of the human body performing the '{}' action in a sentence.",
# Round 2
"""Describe the following body parts actions in short when '{}': head, hand, arm, hip, leg, foot.
Strictly follow these rules:
1. Each body part must be listed separately in this exact order: Head, Hand, Arm, Hip, Leg, Foot
2. Format as "[Body part]: [Description]." with no dashes or numbers
3. Use exactly one line per body part with no blank lines
4. Never combine body parts (e.g., no "Hand & Arm")
5. Keep descriptions brief""",
# Round 3
"""The human body can be further represented as 25 skeleton joints. 
The labels of the joints are: 1-base of the spine, 2-middle of the spine, 3-neck, 4-head, 5-left shoulder, 6-left elbow, 7-left wrist, 8-left hand, 9-right shoulder, 10-right elbow, 11-right wrist, 12-right hand, 13-left hip, 14-left knee, 15-left ankle, 16-left foot, 17-right hip, 18-right knee, 19-right ankle, 20-right foot, 21-spine, 22-tip of the left hand, 23-left thumb, 24-tip of the right hand, and 25-right thumb. 
Based on the movement process of the '{}' action, please point out all joints that are critical for the recognition of this action, including both left and right sides where applicable. 
Strictly follow these rules:
1. List only the numerical labels of critical joints
2. Separate labels with commas and single spaces
3. Order the numbers from smallest to largest
4. Put all numbers on one line with no line breaks
5. Do not include any other text or explanations""",
# Round 4
"""For all the critical joints identified in the previous answer, describe their movement details in short.
Strictly follow these rules:
1. Use exact format: "[label]-[joint name]: [description]."
2. One joint per line, no blank lines
3. List in numerical order (1-25)
4. Exclude non-critical joints
5. If >10 critical joints, select 4-8 most important
6. Keep descriptions brief""",
# Round 5
"""Based on the above information, please summarize the key features that recognize the '{}' action in one sentence.
Strictly follow these rules:
1. No bullet points or dashes
2. No line breaks
3. No joint labels
4. Keep it concise
5. Start directly with the description"""
]

conversation_history = []


def chat_with_LLM(prompt):
    conversation_history.append({"role": "user", "content": prompt})
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=conversation_history
        )
        ai_reply = response.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": ai_reply})
        return ai_reply
    except Exception as e:
        print(f"Error occurred: {e}")
        return None
    

def save_to_file(content, filename):
    question_part, answer_part = content.split("Round", 1)[0], "Round" + content.split("Round", 1)[1]
    processed_answer = answer_part.replace("*", "")
    processed_answer = "\n".join(
        [line if line.strip() != "" or i == 0 else ""
         for i, line in enumerate(processed_answer.split("\n"))]
    )
    final_content = question_part + processed_answer
    with open(filename, "a", encoding="utf-8") as file:
        file.write("\n" + final_content + "\n\n")
        file.write("=" * 80 + "\n")


def process_action(action_name, index):
    global conversation_history
    conversation_history = []
    
    dialogs = [dialog.format(action_name) for dialog in dialogs_template]
    
    global dataset_name
    output_dir = dataset_name
    os.makedirs(output_dir, exist_ok=True)
    safe_filename = f"{index:03d}_{action_name.replace('/', '_').replace(' ', '_')}.txt"
    output_filename = os.path.join(output_dir, safe_filename)
    
    for i, dialog in enumerate(dialogs, 1):
        response = chat_with_LLM(dialog)
        if response is None:
            print(f"Failed to get response for {action_name}, round {i}")
            continue
        
        save_content = f"Round {i} Question:\n{dialog}\n\nRound {i} Answer:\n{response}"
        save_to_file(save_content, output_filename)      


def main():
    print(f"Starting processing for {len(action_names)} actions")
    
    for index, action_name in enumerate(tqdm(action_names, desc="Processing actions"), 1):
        process_action(action_name, index)
    
    print("All actions processed")


if __name__ == "__main__":
    main()
