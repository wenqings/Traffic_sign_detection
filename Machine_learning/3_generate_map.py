import pandas as pd
import json

df = pd.read_csv('D:\\RSA\\output\\train_labels.csv')

type_list = set()
# remove sign version number, eg.information--parking--g1 --> information--parking
for i in range(len(df)):
    if df['class'][i] not in type_list:
        type_list.add(df['class'][i])

print(len(type_list))

def add_item(f,name, id):
    f.write('item {\n')
    f.write('  name : "'+name+'"\n')
    f.write('  id : '+str(id)+'\n')
    f.write('}\n')

map_dick = {}
output_path = 'D:\\RSA\\output\\NA_traffic_sign_map.pbtxt'
output_json = 'D:\\RSA\\output\\NA_traffic_sign_map.json'
f = open(output_path,"w")


type_list = list(type_list)
# Label map id 0 is reserved for the background label, so the id begin at 1
for i in range(len(type_list)):
    add_item(f,type_list[i],i+1)
    map_dick.update({type_list[i]:i+1})
with open(output_json, 'w') as fp:
    json.dump(map_dick,fp,indent=2)
