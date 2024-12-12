import requests

url = 'https://bug-free-computing-machine-vwxj4jpg9qvcx747-8080.app.github.dev/'

data = {'url': 'https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg'}

# Make the request and store the response
response = requests.post(url, json=data)

# Print status code and response content
print(f"Status Code: {response.status_code}")
print(f"Response Content: {response.text}")

# Only try to parse JSON if we got a successful response
if response.status_code == 200:
    result = response.json()
    print(f"JSON Result: {result}")