import torch
import os
from ..clip import clip

import sys
from openai import OpenAI
from pathlib import Path


my_api_key = "$THE_API_KEY$"
my_base_url = "$THE_BASE_URL$"

context_dialog_template = [
# Round 6
"""In skeleton-based action recognition, the '{}' action needs to be distinguished from similar actions like {}.
Please summarize the distinctive motion characteristics of the '{}' action in one sentence, highlighting its key distinguishing features.
Strictly follow these rules:
1. No bullet points or dashes
2. No line breaks
3. No joint labels
4. Keep it concise (<70 tokens)
5. Start directly with the description"""
]


class Interaction():
    
    def __init__(self, num_classes, work_dir):
        self.client = OpenAI(api_key=my_api_key, base_url=my_base_url)
        
        self.num_classes = num_classes
        if num_classes == 60:
            self.dataset_name = 'nturgbd_60'
        elif num_classes == 120:
            self.dataset_name = 'nturgbd_120'
        elif num_classes == 99:
            self.dataset_name = 'finegym'
        elif num_classes == 400:
            self.dataset_name = 'k400'
        elif num_classes == 155:
            self.dataset_name = 'uav_human'
        
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        label_name_path = os.path.join(self.script_dir, "text", "label_map", f"{self.dataset_name}.txt")
        with open(label_name_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            self.action_names = [line.strip() for line in lines if line.strip()]
        
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent.parent
        relative_path = Path(work_dir.lstrip("./"))
        self.work_dir = project_root / relative_path / self.dataset_name
        self.save_work_dir = None
        
        global context_dialog_template
        self.dialog_template = context_dialog_template

    
    def read_file_safely(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                return file.read()
        except FileNotFoundError:
            print(f"File not found: {filepath}")
            return None
        except Exception as e:
            print(f"Error reading file {filepath}: {e}")
            return None

    
    """
    Please note that due to security policies, the GPT model may refuse to respond to certain sensitive terms, i.e., the word 'gun' in the NTU 120 dataset 'A110 shoot at other person with a gun'.
    For training stability, it is recommended to start by using Deepseek.    
    """
    
    def chat_with_LLM(self, llm_model, prompt, conversation_history):
        if llm_model == "gpt4o":
            chat_model = "gpt-4o"
        conversation_history.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=chat_model,
                messages=conversation_history
            )
            ai_reply = response.choices[0].message.content
            conversation_history.append({"role": "assistant", "content": ai_reply})
            return conversation_history
        except Exception as e:
            print(f"Error occurred: {e}")
            return None
    
    
    def save_to_file(self, conversation_history, output_filename):
        with open(output_filename, 'w', encoding='utf-8') as file:
            round_number = 1
            for i in range(0, len(conversation_history), 2):
                if i+1 >= len(conversation_history):
                    break
                user_msg = conversation_history[i]
                assistant_msg = conversation_history[i+1]
                file.write(f"\nRound {round_number} Question:\n")
                file.write(user_msg['content'] + "\n\n")
                file.write(f"Round {round_number} Answer:\n")
                file.write(assistant_msg['content'] + "\n\n")
                file.write("=" * 80 + "\n")
                round_number += 1

    
    def get_description(self, llm_model):
        
        if self.save_work_dir == None:
            text_file_path = os.path.join(self.script_dir, "text", llm_model, self.dataset_name)
        else:
            text_file_path = self.save_work_dir
        
        text_map = []
        for index, action_name in enumerate(self.action_names):
            # Generate the filename
            safe_filename = f"{index+1:03d}_{action_name.replace('/', '_').replace(' ', '_')}.txt"
            filepath = os.path.join(text_file_path, safe_filename)
            content = self.read_file_safely(filepath)
            if content is None:
                continue

            # Split content into rounds
            rounds = content.split("=" * 80 + "\n")
            
            # Process the final answer
            round_end = rounds[-2].strip()
            round_end_answer = round_end.split(f"Round {len(rounds)-1} Answer:\n")[-1].strip()
            text_map.append(round_end_answer)
        
        text_list = []
        for action_description in text_map:
            text_list.append(clip.tokenize(action_description))
        
        return text_list


    def update_description(self, llm_model, class_list):
        
        if self.save_work_dir == None:
            text_file_path = os.path.join(self.script_dir, "text", llm_model, self.dataset_name)
            self.save_work_dir = self.work_dir
            os.makedirs(self.save_work_dir, exist_ok=True)
        else:
            text_file_path = self.save_work_dir

        for index, action_name in enumerate(self.action_names):
            # Generate the filename
            safe_filename = f"{index+1:03d}_{action_name.replace('/', '_').replace(' ', '_')}.txt"
            filepath = os.path.join(text_file_path, safe_filename)
            content = self.read_file_safely(filepath)
            
            # Split content into rounds
            rounds = content.split("=" * 80 + "\n")
            rounds = [round_text.strip() for round_text in rounds if round_text.strip()]
            
            conversation_history = []
            round_count = 0
            for round_text in rounds:
                if not round_text.startswith("Round"):
                    continue
                round_count += 1
                if round_count > 5:
                    break
                # Split into lines and remove empty lines
                lines = [line.strip() for line in round_text.split("\n") if line.strip()]
                # Find question and answer sections
                question_lines = []
                answer_lines = []
                current_section = None
                for line in lines:
                    if line.startswith("Round") and "Question:" in line:
                        current_section = "question"
                        line = line.split("Question:", 1)[1].strip()
                        if line:
                            question_lines.append(line)
                    elif line.startswith("Round") and "Answer:" in line:
                        current_section = "answer"
                        line = line.split("Answer:", 1)[1].strip()
                        if line:
                            answer_lines.append(line)
                    elif current_section == "question":
                        question_lines.append(line)
                    elif current_section == "answer":
                        answer_lines.append(line)
                # Join the lines to form complete question and answer
                question = '\n'.join([line for line in question_lines if line])
                answer = '\n'.join([line for line in answer_lines if line])
                if question and answer:
                    conversation_history.append({"role": "user", "content": question})
                    conversation_history.append({"role": "assistant", "content": answer})

            similar_label = class_list[index]
            if len(similar_label) > 0:
                similar_names = [self.action_names[idx] for idx in similar_label]
                similar_action = ", ".join(f"'{action}'" for action in similar_names)
                dialog = self.dialog_template[0].format(action_name, similar_action, action_name)
                conversation_history = self.chat_with_LLM(llm_model, dialog, conversation_history)
            
            output_filename = os.path.join(self.save_work_dir, safe_filename)
            self.save_to_file(conversation_history, output_filename)
        
        return True
