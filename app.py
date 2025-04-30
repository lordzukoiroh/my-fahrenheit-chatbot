
import torch
import gradio as gr
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from google.colab import drive
from datasets import Dataset
import json
import os
from typing import List, Tuple
import shutil
from functools import partial

# === CSS ve Emoji Fonksiyonu ===
current_css = 