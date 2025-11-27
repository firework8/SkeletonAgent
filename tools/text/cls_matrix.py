import os
import numpy as np
import re


def read_action_names(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        action_names = [line.strip() for line in lines if line.strip()]
    return action_names

def read_file_safely(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None

def find_indices_for_label(label, bone_pairs):
    indices = []
    for index, pair in enumerate(bone_pairs):
        if label in pair:
            indices.append(index)
    return indices


model_name = "gpt4o"
# nturgbd_60  nturgbd_120  finegym  k400  uav_human
dataset_name = "nturgbd_60"
# j b k
modality = 'j'

dataset_path = f"label_map/{dataset_name}.txt"
model_path = os.path.join(model_name, "cls_matrix")


if __name__ == "__main__":
    
    action_names = read_action_names(dataset_path)
    os.makedirs(model_path, exist_ok=True)
    
    
    if dataset_name == 'nturgbd_60':
        point_num = 25
    elif dataset_name == 'nturgbd_120':
        point_num = 25
    elif dataset_name == 'finegym':
        point_num = 20
    elif dataset_name == 'k400':
        point_num = 20
    elif dataset_name == 'uav_human':
        point_num = 17
        
    if point_num == 25:
        bone_pairs = ((0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5), (7, 6), (8, 20), (9, 8),
                    (10, 9), (11, 10), (12, 0), (13, 12), (14, 13), (15, 14), (16, 0), (17, 16), (18, 17),
                    (19, 18), (21, 22), (20, 20), (22, 7), (23, 24), (24, 11))
        k_bone_pairs = ((0, 20), (1, 1), (2, 2), (3, 20), (4, 4), (5, 20), (6, 4), (7, 5), (8, 8), (9, 20),
                    (10, 8), (11, 9), (12, 1), (13, 0), (14, 12), (15, 13), (16, 1), (17, 0), (18, 16),
                    (19, 17), (21, 7), (20, 20), (22, 6), (23, 11), (24, 10))
    elif point_num == 20:
        bone_pairs = ((0, 19), (1, 0), (2, 0), (3, 1), (4, 2), (5, 19), (6, 19), (7, 5), (8, 6), (9, 7), (10, 8),
                    (11, 17), (12, 17), (13, 11), (14, 12), (15, 13), (16, 14), (17, 18), (18, 19), (19, 19))
        k_bone_pairs = ((0, 0), (1, 19), (2, 19), (3, 0), (4, 0), (5, 5), (6, 6), (7, 19), (8, 19), (9, 5), 
                    (10, 6),(11, 18), (12, 18), (13, 17), (14, 17), (15, 11), (16, 12), (17, 19), (18, 18), (19, 19))
    elif point_num == 17:
        bone_pairs = ((0, 0), (1, 1), (2, 1), (3, 2), (4, 1), (5, 4), (6, 5), (7, 1), (8, 7), (9, 8), (10, 7),
                    (11, 10), (12, 11), (13, 0), (14, 0), (15, 13), (16, 14))
        k_bone_pairs = ((0, 0), (1, 1), (2, 2), (3, 1), (4, 4), (5, 1), (6, 4), (7, 7), (8, 1), (9, 7), (10, 1),
                    (11, 4), (12, 10), (13, 13), (14, 14), (15, 0), (16, 0))
    
    """
    To conserve tokens, related joint constraints are not dynamically updated.
    You could also choose to move the code to the interactive function to enable real-time updates.
    """
    
    critical_joints_matrix = np.zeros((len(action_names), point_num, point_num))
    
    for index, action_name in enumerate(action_names):
        # Generate the filename
        safe_filename = f"{index+1:03d}_{action_name.replace('/', '_').replace(' ', '_')}.txt"
        filepath = os.path.join(model_name, dataset_name, safe_filename)
        content = read_file_safely(filepath)
        if content is None:
            continue
        
        # Split content into rounds
        rounds = content.split("=" * 80 + "\n")
        
        # Process Round 4 Answer
        if len(rounds) >= 4:
            round4 = rounds[3].strip()
            round4_answer = round4.split("Round 4 Answer:\n")[-1].strip()
            lines = [line.strip() for line in round4_answer.split('\n') if line.strip()]
            
            # Process for critical_joints.npy
            labels = []
            descriptions = []
            for line in lines:
                # Extract label
                match = re.match(r"^(\d+)-", line)
                if match:
                    label = int(match.group(1))
                    labels.append(label)
                    # Set all values in this row to 1 (1-based to 0-based)
                    if modality == 'j':
                        critical_joints_matrix[index, label-1, :] = 1
                    elif modality == 'b':
                        result = find_indices_for_label(label-1, bone_pairs)
                        critical_joints_matrix[index, result, :] = 1
                    elif modality == 'k':
                        result = find_indices_for_label(label-1, k_bone_pairs)
                        critical_joints_matrix[index, result, :] = 1

    # Save the critical joints matrix
    np.save(os.path.join(model_path, f"{dataset_name}_{modality}_critical_joints.npy"), critical_joints_matrix)
    print("Processing complete!")
    