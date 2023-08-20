import requests
import json
import numpy
import numpy as np

#API_KEY = "R4hdgNE2iehi6hh5Oo0eQn78"
#SECRET_KEY = "q9owq4LacXsUDbCwQPta1Pow8lVDo1KR"
access_token = '24.e56fc44edb2a188019d8b36bd9d064a6.2592000.1694164088.282335-37426557'

url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/table_infer/37426557"

image_data = np.loadtxt("/Users/wuzhenghan/Downloads/20181124/image.txt")
test_data = np.loadtxt("/Users/wuzhenghan/Downloads/20181124/testdata1.txt")

test_data = test_data[1:999, :]
test_x = test_data[:, 3:11]
test_y = test_data[:, 2]

imageX = image_data[:, 0:5]

headers = {
    'Content-Type': 'application/json'
}
params = {
    'access_token': access_token
}

# Split test_data into batches of size 100
batch_size = 100

test_data1=image_data[0:120000, :]
test_data2=image_data[30000:60000, :]
test_data3=image_data[60000:90000, :]
test_data4=image_data[90000:120000, :]
test_data5=image_data[120000:147016, :]

batches = [test_data[i:i + batch_size] for i in range(0, len(test_data), batch_size)]

# Initialize the list to store col3_values
col3_values = []

for batch in batches:
    data = {
        "include_req": False,
        "data": []
    }

    for i in range(batch.shape[0]):
        row = {
            "dep": batch[i, 2],
            "r1": batch[i, 3],
            "r2": batch[i, 4],
            "r3": batch[i, 5],
            "r4": batch[i, 6],
            "r5": batch[i, 7],
            "r6": batch[i, 8],
            "r8": batch[i, 9],
            "r10": batch[i, 10]
        }
        data["data"].append(row)

    response = requests.post(url, headers=headers, params=params, data=json.dumps(data))
    result_dict = json.loads(response.text)

    # Extract col3 values from the current batch and append to col3_values  list
    col3_values += [float(result_dict['batch_result'][i]['dep']) for i in range(len(result_dict['batch_result']))]

print(col3_values)

numpy.savetxt("/Users/wuzhenghan/Downloads/20181124/testdataresult51.txt", col3_values)