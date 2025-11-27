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

dataset_name = "finegym"
action_names = read_action_names(f"label_map/{dataset_name}.txt")

dialogs_template = [
# Round 1
"""Describe in full detail the movement of the human body during the '{}' gymnastics element in one sentence.
Focus specifically on the timing, direction, amplitude, and transitions of body parts.
Highlight how the action unfolds from start to finish.
Strictly follow these rules:
1. No line breaks
2. Keep descriptions brief
3. Start directly with the description""",
# Round 2
"""For the '{}' gymnastics element, give a precise and brief description of how each of the following body parts moves: head, hand, arm, hip, leg, foot.
Focus on posture, timing, rotation, and symmetry.
Mention how the motion of each part helps distinguish this element from similar ones.
Strictly follow these rules:
1. Each body part must be listed separately in this exact order: Head, Hand, Arm, Hip, Leg, Foot
2. Format as "[Body part]: [Description]." with no dashes or numbers
3. Use exactly one line per body part with no blank lines
4. Never combine body parts (e.g., no "Hand & Arm")
5. Include directional terms (e.g., forward, upward), angles (e.g., bent at 90°), or timing cues (e.g., early lift, delayed landing)
6. Focus on what makes the motion *unique* in contrast to visually similar actions
7. Keep descriptions brief""",
# Round 3
"""We represent the human body with 20 joints labeled:
1-nose, 2-left eye, 3-right eye, 4-left ear, 5-right ear, 6-left shoulder, 7-right shoulder, 8-left elbow, 9-right elbow, 10-left wrist, 11-right wrist, 12-left hip, 13-right hip, 14-left knee, 15-right knee, 16-left ankle, 17-right ankle, 18-mid hip, 19-thorax, 20-mid shoulder.
Based on the full-body motion of '{}', list the joints that are **most critical** to distinguish this element from other similar elements in the same set.
Include both sides (left/right) when relevant.
Strictly follow these rules:
1. List only the numerical labels of critical joints
2. Separate labels with commas and single spaces
3. Order the numbers from smallest to largest
4. Put all numbers on one line with no line breaks
5. Do not include any other text or explanations""",
# Round 4
"""Now describe the fine-grained motion or alignment of the critical joints listed above **during the '{}' gymnastics element**.
Strictly follow these rules:
1. Use exact format: "[label]-[joint name]: [description]."
2. One line per joint, no blank lines
3. Order by joint number
4. Exclude non-critical joints
5. If >10 critical joints, select 4-8 most important joints for distinguishing this element
6. Include temporal cues (e.g., 'initiates early', 'holds position longer'), angular features (e.g., 'extends beyond 120°'), or asymmetries
7. Keep descriptions brief""",
# Round 5
"""Based on the above information, please summarize how the '{}' gymnastics element can be recognized and distinguished from similar elements in one concise sentence, focusing on coordination, timing, rotation, and posture.
Strictly follow these rules:
1. No joint numbers or bullet points or dashes
2. Use only one sentence, no line breaks
3. Begin immediately with the description (no "The element is...")
4. Emphasize what makes this element visually or structurally different from other similar elements in the same set
5. Keep it concise (<70 tokens)"""
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
    save_filename = f"{index:03d}_{action_name.replace('/', '_').replace(' ', '_')}.txt"
    output_filename = os.path.join(output_dir, save_filename)
    
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
