import requests


BASE_URL = "http://127.0.0.1:8000"


def main():
    """Send GET and POST requests to the local FastAPI app."""
    get_response = requests.get(
        f"{BASE_URL}/",
        timeout=10,
    )

    print(f"Status Code: {get_response.status_code}")
    print(f"Result: {get_response.json()}")

    data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }

    post_response = requests.post(
        f"{BASE_URL}/data/",
        json=data,
        timeout=10,
    )

    print(f"Status Code: {post_response.status_code}")
    print(f"Result: {post_response.json()}")


if __name__ == "__main__":
    main()
