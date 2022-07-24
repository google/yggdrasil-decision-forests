#!/bin/bash

curl -d '{"model_id": "model", "age": 35.0, "education": "HS-grad"}' -H 'Content-Type: application/json' localhost:8081/predict