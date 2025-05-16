import base64
from openai import OpenAI
from io import BytesIO


class LLM:
    def __init__(self, base_url, api_key, llm_model):
        self.base_url = base_url
        self.api_key = api_key
        self.llm_model = llm_model

    def __call__(self, prompt):
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                }
            ],
            model=self.llm_model,
        )
        return chat_completion.choices[0].message.content


class VLM:
    def __init__(self, base_url, api_key, vlm_model):
        self.base_url = base_url
        self.api_key = api_key
        self.vlm_model = vlm_model

    def __call__(self, prompt, image):
        buffered = BytesIO()
        image.save(buffered, format='PNG')
        image_bytes = base64.b64encode(buffered.getvalue())
        image_str = str(image_bytes, 'utf-8')

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    'role': 'user',
                    'content': [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": "data:image/png;base64," + image_str}
                    ]
                }
            ],
            model=self.vlm_model,
        )
        return chat_completion.choices[0].message.content