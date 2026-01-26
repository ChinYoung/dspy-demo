# glm4_dspy.py

import dspy


Lm_Glm = dspy.LM(
    model="zai/glm-4.7",
)


Lm_Ollama_Qwen3_4b = dspy.LM(
    model="ollama/qwen3:4b",
    temperature=0.1,
    ollama_base_url="http://192.168.50.185:11434",
)
