import requests

# Health check
response = requests.get("http://localhost:8000/health")
assert response.status_code == 200
print("Health check passed")

print("Smoke test completed successfully")