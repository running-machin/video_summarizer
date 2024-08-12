import pandas as pd
import pickle
import os
from pycocotools.coco import COCO

def convert_gender(gender_tuple =None):
    result = []
    if gender_tuple is None:
        return result
    # Append male identifiers
    for i in range(1, gender_tuple[0] + 1):
        result.append(f"M_{i}")
    # Append female identifiers
    for i in range(1, gender_tuple[1] + 1):
        result.append(f"W_{i}")
    return result
# import the dataset
dataset_path = '/mnt/g/Github/video_summarizer/datasets/video/dataset'
# videos  = [video for video in os.listdir(data_path) if video.startswith('video')]

df = pd.DataFrame(columns=['Video','Frame','Face'])
# for video in videos:
    # dataset_path = os.path.join(data_path, video)
frames_path = os.path.join(dataset_path, 'frames')
frames = [frame for frame in os.listdir(frames_path) if frame.endswith('.jpg')]
frames_count = len(frames) 
print(len(frames))
# Load the data
file_path = os.path.join(dataset_path, 'labels', 'gender.data')

with open(file_path, 'rb') as f:
    gender_data = pickle.load(f)

# flattern the list of dictionaries to one dictionary
gender_data = {list(item.keys())[0]: list(item.values())[0] for item in gender_data}
print(len(gender_data))
# import the Coco
coco_path = os.path.join(dataset_path, 'labels', 'coco128.json')
coco =COCO(coco_path)
image_mapping = {}  # Mapping of old image IDs to new image IDs
for image in coco.dataset['images']:
    image_mapping[image['file_name']] = image['id']

# image_mapping_inv = {v:k for k,v in image_mapping.items()}
# Create a DataFrame
print(len(image_mapping))

# for item in gender_data:
#     img_id = list(item.keys())[0]
#     gender = list(item.values())[0] #gender(man,women)
#     frame = image_mapping[img_id][0].split('.')[0]
#     # covert gender from tuple to list, eg: (0,1) to [F_1], (1,1) to [M_1, F_1], (0,0) to [], (1,0) to [M_1], (0,2) to[W_1, W_2] , (2,0) to [M_1, M_2], (2,3) to [M_1, M_2, W_1, W_2, W_3]
#     face = convert_gender(gender)    

#     df = df._append({'Video':video,'Frame':frame,'Face':face},ignore_index=True)
#     break
for frame in frames:
    face = convert_gender(gender_data.get(image_mapping.get(frame)))
    video = frame.split('_')[1]
    df = df._append({'Video':video,'Frame':frame.split('_')[2].split('.')[0],'Face':face},ignore_index=True)

# print(df.head())
df.to_csv(os.path.join(dataset_path,'fair_tvsum.csv'),index=False) 