import os
import glob
import pandas as pd
import json


def json_to_csv(path):
    json_list = []
    for json_file_path in glob.glob(path + '/*.json'):
        print(json_file_path)
        with open(json_file_path) as json_file:
            tree = json.load(json_file)

            for i in range(len(tree['objects'])):

                value = (json_file_path.replace('/','\\').split('\\')[-1].split('.')[0]+'.jpg',
                         tree['width'],
                         tree['height'],
                         tree['objects'][i]['label'].replace('--g1','').replace('--g2','').replace('--g3','').replace('--g4','').replace('--g5','').replace('--g6',''),
                         int(tree['objects'][i]['bbox']['xmin']),
                         int(tree['objects'][i]['bbox']['ymin']),
                         int(tree['objects'][i]['bbox']['xmax']),
                         int(tree['objects'][i]['bbox']['ymax'])
                         )
                json_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    json_df = pd.DataFrame(json_list, columns=column_name)
    return json_df


def main():

    json_df = json_to_csv('D:\\MTSD\\train')
    json_df.to_csv('D:\\MTSD\\train_labels.csv', index=None)

    json_df = json_to_csv('D:\\MTSD\\test')
    json_df.to_csv('D:\\MTSD\\test_labels.csv', index=None)


main()
