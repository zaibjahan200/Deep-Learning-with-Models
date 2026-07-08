# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Structure

This repository contains several independent deep learning projects:

- `violence-detection/`: Contains blood detection and segmentation tools.
    - `blood-segment/`: A FastAPI application for blood segmentation and blurring using a Keras model (`best_model.keras`).
    - `blood_api/`: Another FastAPI implementation for blood detection, using a different model (`phase2_checkpoint_best.keras`).
- `fashion_MNIST_app_with_model/`: A simple apparel classification Streamlit app.
- `tweet_classifier/`: A set of notebooks and a FastAPI app for classifying tweets using embeddings.

## Development Commands

### Running FastAPI Apps
The API applications in this repo typically use `uvicorn` to serve the API.

To run the `blood-segment` API:
```bash
# From the root directory
uvicorn violence-detection.blood-segment.app.main:app --reload
```

To run the `blood_api` API:
```bash
# From the violence-detection/blood_api directory
python app.py
```

### Dependency Management
Each sub-project contains its own `requirements.txt`. Ensure you are in the correct directory when installing dependencies.

```bash
pip install -r violence-detection/blood-segment/requirements.txt
# or
pip install -r violence-detection/blood_api/requirements.txt
```

## Architectural Notes

- The `violence-detection` sub-projects (`blood-segment` vs `blood_api`) demonstrate two different approaches to blood detection:
    - `blood-segment` performs image segmentation using a mask, which allows it to blur only the detected blood regions precisely.
    - `blood_api` uses a hybrid classification and regression model to provide a bounding box, which it then blurs.
- Both models rely on Keras/TensorFlow and OpenCV. Ensure your environment is configured for the appropriate machine learning library versions as specified in each project's `requirements.txt`.
