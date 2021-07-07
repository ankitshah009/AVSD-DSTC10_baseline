import json
import pandas as pd
import pdb
import sys

def read_json(json_file):
    f=open(json_file)
    data=json.loads(f.read())
    return data

def generate_stats(train_json, test_json, val_json, Charades_train_csv, Charades_test_csv):
    train_data = read_json(train_json)
    test_data = read_json(test_json)
    val_data = read_json(val_json)
    train_df = pd.read_csv(Charades_train_csv)
    test_df = pd.read_csv(Charades_test_csv)


    char_train_ids = train_df['id'].values.tolist()
    char_test_ids = test_df['id'].values.tolist()

    avsd_train_id = [item['image_id'] for item in train_data['dialogs']]
    avsd_test_id = [item['image_id'] for item in test_data['dialogs']]
    avsd_val_id = [item['image_id'] for item in val_data['dialogs']]


    print("Num AVSD train files ", len(train_data['dialogs']))
    print("Num AVSD test files ", len(test_data['dialogs']))
    print("Num AVSD val files ", len(val_data['dialogs']))
    
    print("Num Charades files train ", len(train_df))
    print("Num Charades files test ", len(test_df))

    print("Num Charades train file with Action ", len(train_df) - train_df['actions'].isna().sum())
    print("Num Charades train file without Action ", train_df['actions'].isna().sum())

    print("Num Charades test file with Action ", len(test_df) - test_df['actions'].isna().sum())
    print("Num Charades test file without Action ", test_df['actions'].isna().sum())

    
    print("Num of AVSD train in Charades train ", len(set(char_train_ids).intersection(avsd_train_id)))
    print("Num of AVSD val in Charades train ", len(set(char_train_ids).intersection(avsd_val_id)))
    print("Num of AVSD test in Charades train ", len(set(char_train_ids).intersection(avsd_test_id)))

    print("Num of AVSD train in Charades test ", len(set(char_test_ids).intersection(avsd_train_id)))
    print("Num of AVSD val in Charades test ", len(set(char_test_ids).intersection(avsd_val_id)))
    print("Num of AVSD test in Charades test ", len(set(char_test_ids).intersection(avsd_test_id)))

    no_action_charades_train_ids = train_df[train_df['actions'].isna()]['id'].values.tolist()
    action_charades_train_ids = train_df[~train_df['actions'].isna()]['id'].values.tolist()


    no_action_charades_test_ids = test_df[test_df['actions'].isna()]['id'].values.tolist()
    action_charades_test_ids = test_df[~test_df['actions'].isna()]['id'].values.tolist()

    print("Num AVSD train with Action ", len(set(avsd_train_id).intersection(action_charades_train_ids)))
    print("Num AVSD train without Action ", len(set(avsd_train_id).intersection(no_action_charades_train_ids)))

    print("Num AVSD test with Action ", len(set(avsd_test_id).intersection(action_charades_test_ids)))
    print("Num AVSD test without Action ", len(set(avsd_test_id).intersection(no_action_charades_test_ids)))
    print("Num AVSD val with Action ", len(set(avsd_val_id).intersection(action_charades_test_ids)))
    print("Num AVSD val without Action ", len(set(avsd_val_id).intersection(no_action_charades_test_ids)))
    
    pdb.set_trace()
    pdb.set_trace()
    

if __name__=="__main__":
    generate_stats(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])

    """
    Sample command - python generate_stats.py train_set4DSTC7-AVSD.json test_set4DSTC7-AVSD.json valid_set4DSTC7-AVSD.json Charades_v1_train.csv Charades_v1_test.csv
    """
