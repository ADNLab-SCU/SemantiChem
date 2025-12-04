import time
import requests
import json
import csv


every_csv_n=600
final_run=2000
# default URL
url = "http://0.0.0.0:8000/v1/chat/completions"

headers = {
    "accept": "application/json",
    "Content-Type": "application/json",
}

data_1 = {
    "model": "string",
    "messages": [
        {
            "role": "user",
            "content": "Please show me a ligand which haven't been reported. Be sure the structure is novel and unique."
        }
    ],
    "tools": [],
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 0.9,
    "n": 1,
    "max_tokens": 2048,
    "stream": False



s_time = time.time()
replies1=[]

print("start request.")
for i in range(every_csv_n):
    response = requests.post(url, headers=headers, data=json.dumps(data_1))
    print((time.time()-s_time))
    reply=response.json()
    replies1.append(reply['choices'][0]['message']['content'])
    print(str(reply['choices'][0]['message']['content']))
    print("finishing:",i)
with open('result.csv',mode='w',newline='',encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Id','Selfies'])
    for index, reply in enumerate(replies1, start=1):
        writer.writerow([index, reply])
print("finsh.")