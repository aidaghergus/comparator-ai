# uv pip install instructor ollama pydantic python-dotenv openai

import instructor
from openai import OpenAI
from typing import Optional, List
from enum import Enum
from pydantic import BaseModel, Field


# ==============================================================================
# MODELS
# ==============================================================================

class AuthorInfo(BaseModel):
    name: Optional[str] = Field(description="The name of the author")
    verified_buyer: bool = Field(description="If the buyer is verified or not")
    experience_level: Optional[str] = None


class RecommendationStatus(Enum):
    recommended = "recommended"
    not_recommended = "not_recommended"
    neutral = "neutral"
    not_specified = "not_specified"


class ReviewAnalysis(BaseModel):
    product_name: str
    rating: int = Field(ge=1, le=5)
    author: AuthorInfo
    pros: List[str] = []
    cons: List[str] = []
    is_authentic: Optional[bool] = None
    summary: str = Field(description="If is human generated or not")
    would_recommend: RecommendationStatus = RecommendationStatus.not_specified
    confidence_score: int = Field(ge=0, le=10, description="0 means no confidence, 100 means full confidence")


# ==============================================================================
# SERVICE
# ==============================================================================

class ReviewAnalyzerService:
    def __init__(self):
        self.client = instructor.from_openai(
            OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama"
            ),
            mode=instructor.Mode.JSON
        )
        self.model_name = "qwen3:0.6b"

    def analyze(self, text: str) -> ReviewAnalysis:
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ești un analizator de recenzii. "
                        "Extragi date structurat. "
                        "Returnează DOAR JSON valid."
                    )
                },
                {
                    "role": "user",
                    "content": f"Analizează această recenzie:\n\n{text}"
                }
            ],
            temperature=0,
            max_tokens=800,
            response_model=ReviewAnalysis,
            max_retries=3
        )


# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================

if __name__ == "__main__":

    review_text = "Am cumparat Laptopul Dell XPS 13 acum 2 saptamani si sunt foarte multumit! " \
    "PRO: Ecran superb 4K, bateria tine 10 ore, foarte usor si portabil. " \
    "CON: Pret cam mare, nu are port USB-A. Dau 4 stele si il recomand! " \
    "-- Andrei Popescu, utilizator avansat"

    if len(review_text.strip()) < 10:
        print("Text prea scurt.")
        exit()

    service = ReviewAnalyzerService()

    try:
        result = service.analyze(review_text)

        print("\n=== REZULTAT ===")
        print(result.model_dump_json(indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"\nEroare: {e}")