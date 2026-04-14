import requests
import json

def test_chat():
    url = "http://localhost:8000/api/chat"
    data = {"message": "Hello, how are you?"}
    
    print(f"Sending request to {url}...")
    try:
        response = requests.post(url, data=data, stream=True)
        print(f"Status Code: {response.status_code}")
        
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data: "):
                    content = json.loads(decoded_line[6:])
                    print(f"Received: {content}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_chat()
