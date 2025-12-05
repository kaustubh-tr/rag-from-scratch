import asyncio
from typing import Optional
from openai import AsyncOpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from src.rag.core.interfaces import BaseLLM
from config.settings import Config


class OpenAILLM(BaseLLM):
    def __init__(self, model_name: str = Config.OPENAI_LLM_MODEL):
        super().__init__(model_name)
        self.client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)

    async def generate(self, query: str, context: str = "") -> str:
        response = await self.client.responses.create(
            model=self.model_name,
            input=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_prompt.format(query=query, context=context)},
            ],
            temperature=0,
            max_output_tokens=512
        )
        return response.output_text


class HuggingFaceLLM(BaseLLM):
    def __init__(self, model_name: str = Config.HF_LLM_MODEL):
        super().__init__(model_name)
        self.pipe = pipeline(
            "text-generation",
            model=self.model_name,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )

    async def generate(self, query: str, context: str = "") -> str:
        prompt = self.system_prompt + "\n" + self.user_prompt.format(query=query, context=context)
        # Run blocking generation in thread
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.pipe(
                text_inputs=prompt, max_new_tokens=512, do_sample=True, temperature=0.1
            ),
        )
        # We can either use do_sample=False and temperature=0.0 or do_sample=True and temperature>0.0
        # If we use do_sample=True and temperature=0.0, it will throw an error
        generated_text = result[0]["generated_text"]
        if generated_text.startswith(prompt):
            return generated_text[len(prompt):].strip()
        return generated_text
