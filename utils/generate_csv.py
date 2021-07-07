import os
import sys
import pdb
import json
import re
import pandas as pd

def generate_csv(duration_file,inp_json,phase,output_csv):
    df1=pd.read_csv(duration_file,sep=' ',header=None)
    f=open(inp_json,'r')
    d1=json.loads(f.read())

    column_names=['video_id','caption','start','end','duration','phase','idx']
    df2=pd.DataFrame(columns = column_names, dtype=object)
    ld={item1['image_id'] : item1 for item1 in d1['dialogs']}
    d_list=[]
    c=0
    for index,item in df1.iterrows():
        key = re.sub(r'\.mp4$', '', item[0])
        if key in ld:
            item1 = ld[key]
            d={}
            d['video_id']=key
            d['idx']=c
            d['duration']=item[1]
            d['end'] = item[1]
            d['start'] = 0
            d['caption'] = ' '.join(['Q: ' + item_ins['question']+ ' A: ' + item_ins['answer'] for item_ins in item1['dialog']])
            d['phase'] = phase
            d_list.append(d)
            c+=1 

    df2 = pd.DataFrame(d_list)
    df2 = df2.reindex(columns=column_names)
    df2.to_csv(output_csv,sep='\t',index=None)
    #pdb.set_trace()

if __name__=="__main__":
    generate_csv(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])
