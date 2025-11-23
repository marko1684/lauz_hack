#!/bin/bash
export EMBEDDINGS_PATH="/home/marko/Documents/lauz_hack/scraper/full_embeddings_finetuned.pkl"
export FAISS_PATH="/home/marko/Documents/lauz_hack/scraper/full_embeddings_finetuned.faiss"
export MODEL_PATH="/home/marko/Documents/lauz_hack/scraper/patent-finetuned-model/final"
cd backend
python3 api.py
