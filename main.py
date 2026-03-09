"""
Product Comparison Engine cu Instructor + pipeline Generator → Verificator → Retry.
Garantează output structurat validat Pydantic prin Instructor.

Pipeline în 2 pași:
  1. Generator: LLM oferă răspuns cu GÂNDIRE + RĂSPUNS + confidence score
  2. Verificator: Al doilea LLM evaluează validitatea logicii (da/nu/nesigur)
  3. Retry: Dacă e respins, se retrimite cu feedback (max 3 încercări)
"""

import hashlib
import os
from typing import List, Optional
from dotenv import load_dotenv
import instructor
import openai
from diskcache import Cache
from fastapi import FastAPI, HTTPException
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import html2text
from pydantic import BaseModel, Field, field_validator
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURARE
# =============================================================================

cache = Cache(directory=os.getenv("CACHE_DIR", "./cache"))

client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
)

# Patch cu Instructor pentru structured outputs
instructor_client = instructor.from_openai(client, mode=instructor.Mode.JSON)

MODEL = "llama-3.3-70b-versatile"

MAX_RETRIES = 3  # Numărul maxim de reîncercări în pipeline


# =============================================================================
# MODELE PYDANTIC
# =============================================================================

class ProductData(BaseModel):
    """Date extrase despre produs."""
    titlu: str = Field(description="Numele produsului")
    descriere: str = Field(description="Descriere scurtă")
    specificatii: str = Field(description="Specificații tehnice cheie")
    preț: str = Field(default="")
    extras_din: str = Field(description="'scraping' sau 'text'")


class FeatureComparison(BaseModel):
    """O linie din tabelul comparativ."""
    feature_name: str = Field(description="Numele caracteristicii")
    produs_a_value: str = Field(description="Valoare produs A")
    produs_b_value: str = Field(description="Valoare produs B")
    rationale: str = Field(description="Analiza diferentelor si ce conteaza in luarea deciziei")
    winner_score: int = Field(ge=1, le=10, description="Diferență 1-10")
    winner: str = Field(pattern="^(A|B|Egal)$")
    relevant_pentru_user: bool


class Verdict(BaseModel):
    """Verdict final al comparației."""
    câștigător: str = Field(pattern="^(A|B|Egal)$")
    scor_a: int = Field(ge=0, le=100, description="Scorul pentru primul produs")
    scor_b: int = Field(ge=0, le=100, description="Scorul pentru al doilea produs")
    diferență_semificativă: bool = Field(description="Daca exista o diferenta mare intre produse")
    argument_principal: str = Field(max_length=500)
    compromisuri: str = Field(max_length=500)


# ---------------------------------------------------------------------------
# MODELE NOI: Raționament explicit + Confidence + Verificator
# ---------------------------------------------------------------------------

class ReasonedComparisonResult(BaseModel):
    """
    Model Generator: include GÂNDIRE explicită + RĂSPUNS + scor de încredere.
    Instructor forțează structura la fiecare generare.
    """
    # --- Raționament explicit (Chain-of-Thought) ---
    gandire: str = Field(
        description=(
            "Pași de raționament expliciți ÎNAINTE de concluzie. "
            "Formatul: 'GÂNDIRE: [pas1] → [pas2] → [pas3]'. "
            "Trebuie să acopere: analiza preferințelor userului, "
            "maparea specificațiilor, identificarea trade-off-urilor."
        )
    )

    # --- Scor de încredere al Generatorului ---
    confidence: float = Field(
        ge=0.0, le=1.0,
        description=(
            "Scorul de încredere al Generatorului în propria analiză. "
            "0.0 = complet nesigur, 1.0 = certitudine maximă. "
            "Bazat pe: completitudinea datelor, claritatea preferințelor, "
            "și consistența logicii interne."
        )
    )
    confidence_rationale: str = Field(
        max_length=300,
        description="Explicație scurtă pentru scorul de încredere ales."
    )

    # --- Rezultat comparație (RĂSPUNS) ---
    produs_a_titlu: str = Field(description="Titlu produs A")
    produs_b_titlu: str = Field(description="Titlu produs B")
    features: List[FeatureComparison] = Field(description="Tabel comparativ")
    verdict: Verdict
    preferinte_procesate: str = Field(description="Rezumat preferințe user")


class VerificationResult(BaseModel):
    """
    Model Verificator: evaluează validitatea logicii din ReasonedComparisonResult.
    Returnează decizie + motiv + analiza confidence-ului Generatorului.
    """
    decizie: str = Field(
        pattern="^(da|nu|nesigur)$",
        description="'da' = logică validă, 'nu' = respins, 'nesigur' = necesită clarificare"
    )
    motiv: str = Field(
        max_length=600,
        description=(
            "Motivul deciziei. OBLIGATORIU dacă decizie='nu' sau 'nesigur'. "
            "Trebuie să fie specific și acționabil pentru retry."
        )
    )
    probleme_identificate: List[str] = Field(
        default_factory=list,
        description="Listă de probleme specifice găsite în raționament (goală dacă decizie='da')"
    )
    # Analiza confidence-ului Generatorului
    confidence_assessment: str = Field(
        max_length=400,
        description=(
            "Evaluarea Verificatorului față de scorul de confidence al Generatorului. "
            "Este confidence-ul justificat? Subevaluat sau supraevaluat?"
        )
    )
    confidence_adjusted: float = Field(
        ge=0.0, le=1.0,
        description=(
            "Scorul de confidence AJUSTAT de Verificator după analiza logicii. "
            "Poate diferi de confidence-ul original al Generatorului."
        )
    )
    feedback_pentru_retry: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Instrucțiuni specifice pentru Generator dacă trebuie să reîncerce."
    )


class FinalComparisonResult(BaseModel):
    """
    Model final returnat de API: include rezultatul + metadata pipeline.
    """
    # Rezultatul comparației
    produs_a_titlu: str
    produs_b_titlu: str
    features: List[FeatureComparison]
    verdict: Verdict
    preferinte_procesate: str

    # Metadata pipeline
    gandire: str = Field(description="Raționamentul explicit al Generatorului")
    confidence_generator: float = Field(description="Confidence-ul original al Generatorului")
    confidence_verificator: float = Field(description="Confidence-ul ajustat de Verificator")
    confidence_rationale: str
    verificare_decizie: str = Field(description="Decizia Verificatorului: da/nu/nesigur")
    numar_incercari: int = Field(description="Câte încercări au fost necesare")
    pipeline_log: List[str] = Field(description="Log-ul pașilor din pipeline")


class ProductInput(BaseModel):
    sursa: str = Field(..., min_length=3)
    este_url: bool = Field(default=False)

    class Config:
        json_schema_extra = {
            "example": {
                "sursa": "iPhone 15: A16, 6GB RAM, 48MP camera",
                "este_url": False
            }
        }


class ComparisonRequest(BaseModel):
    produs_a: ProductInput
    produs_b: ProductInput
    preferinte: str = Field(..., min_length=5, max_length=1000)
    buget_maxim: Optional[int] = Field(None, ge=100)


# =============================================================================
# SCRAPING
# =============================================================================

async def scrape_product(url: str) -> ProductData:
    """
    Scrapează orice pagină de produs cu BeautifulSoup.
    Elimină elemente inutile, păstrează tot conținutul relevant.
    """
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-setuid-sandbox']
            )
            page = await browser.new_page(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            )

            await page.goto(url, wait_until="networkidle", timeout=25000)
            await page.wait_for_timeout(2000)

            html = await page.content()
            title = await page.title()
            await browser.close()

            soup = BeautifulSoup(html, 'html.parser')

            for tag in soup.find_all([
                'script', 'style', 'nav', 'footer', 'header',
                'aside', 'noscript', 'iframe', 'svg', 'canvas',
                'button', 'input', 'form', 'select', 'textarea',
            ]):
                tag.decompose()

            content_parts = []

            h1 = soup.find('h1')
            if h1:
                product_title = h1.get_text(strip=True)
                if product_title:
                    content_parts.append(f"PRODUCT: {product_title}")

            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc and meta_desc.get('content'):
                content_parts.append(f"DESCRIPTION: {meta_desc['content'][:500]}")

            for p in soup.find_all('p'):
                text = p.get_text(strip=True)
                if len(text) > 30:
                    content_parts.append(text)

            for ul in soup.find_all(['ul', 'ol']):
                items = []
                for li in ul.find_all('li'):
                    item_text = li.get_text(strip=True)
                    if len(item_text) > 5:
                        items.append(item_text)
                if items:
                    content_parts.append(" | ".join(items[:15]))

            for table in soup.find_all('table'):
                rows = []
                for tr in table.find_all('tr')[:25]:
                    row_cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                    if row_cells and any(cell for cell in row_cells):
                        rows.append(": ".join(row_cells[:2]))
                if rows:
                    content_parts.append("SPECS: " + " | ".join(rows[:10]))

            for element in soup.find_all(['div', 'section', 'article', 'main']):
                text = element.get_text(strip=True)
                if 200 < len(text) < 2000:
                    is_new = True
                    for existing in content_parts[-5:]:
                        if text[:100] in existing or existing[:100] in text:
                            is_new = False
                            break
                    if is_new:
                        content_parts.append(text[:800])

            seen_fragments = set()
            final_content = []

            for part in content_parts:
                normalized = " ".join(part.lower().split())[:100]
                if normalized not in seen_fragments and len(part) > 20:
                    seen_fragments.add(normalized)
                    final_content.append(part)

            full_text = "\n\n".join(final_content[:40])

            price = ""
            price_indicators = ['price', 'pret', 'preț', 'cost', '€', '$', 'lei', 'ron']
            for part in final_content[:10]:
                lower = part.lower()
                if any(ind in lower for ind in price_indicators):
                    import re
                    matches = re.findall(r'[\d\s.,]+(?:lei|ron|€|\$|eur|usd)?', part, re.IGNORECASE)
                    if matches:
                        price = " ".join(matches[:2])
                        break

            return ProductData(
                titlu=title[:300] if title else url.split('/')[-1][:50],
                descriere=full_text[:6000],
                specificatii="",
                preț=price[:100],
                extras_din="beautifulsoup_clean"
            )

    except Exception as e:
        raise HTTPException(422, f"Scraping failed: {str(e)}")


def parse_text_input(text: str) -> ProductData:
    """Parsează input text liber."""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    return ProductData(
        titlu=lines[0][:200] if lines else "Unknown",
        descriere='\n'.join(lines[:20]),
        specificatii="",
        preț="",
        extras_din="text"
    )


# =============================================================================
# PIPELINE: GENERATOR
# =============================================================================

def _build_generator_prompt(
        prod_a: ProductData,
        prod_b: ProductData,
        preferinte: str,
        feedback_anterior: Optional[str] = None,
        incercare: int = 1
) -> tuple[str, str]:
    """Construiește prompt-urile pentru Generator."""

    system_prompt = """Ești un expert în compararea produselor cu raționament explicit.

INSTRUCȚIUNI OBLIGATORII:
1. Câmpul 'gandire' trebuie să conțină raționamentul tău pas cu pas ÎNAINTE de concluzie.
   Format: 'GÂNDIRE: [analiză preferințe] → [mapare specificații] → [identificare trade-off-uri] → [justificare verdict]'
2. Câmpul 'confidence' (0.0-1.0) reflectă cât de sigur ești pe analiză:
   - Scade dacă datele sunt incomplete, vagi sau contradictorii
   - Crește dacă specificațiile sunt clare și preferințele sunt precise
3. Fii precis cu specificațiile tehnice. Nu inventa date lipsă.
4. Câștigătorul trebuie determinat STRICT pe preferințele userului."""

    retry_context = ""
    if feedback_anterior and incercare > 1:
        retry_context = f"""
⚠️ ÎNCERCAREA {incercare}/3 - FEEDBACK DIN VERIFICARE ANTERIOARĂ:
{feedback_anterior}

Adresează EXPLICIT aceste probleme în câmpul 'gandire' și corectează analiza.
"""

    user_prompt = f"""{retry_context}Compară aceste produse pentru userul care vrea: "{preferinte}"

PRODUS A: {prod_a.titlu}
Descriere: {prod_a.descriere[:6000]}
Spec: {prod_a.specificatii[:4000]}

PRODUS B: {prod_b.titlu}
Descriere: {prod_b.descriere[:6000]}
Spec: {prod_b.specificatii[:4000]}

Generează tabelul comparativ cu DOAR feature-urile relevante pentru preferințele userului.
Completează câmpul 'gandire' cu raționamentul tău explicit înainte de concluzie.
Evaluează-ți confidence-ul sincer bazat pe calitatea datelor disponibile."""

    return system_prompt, user_prompt


async def generator_step(
        prod_a: ProductData,
        prod_b: ProductData,
        preferinte: str,
        feedback_anterior: Optional[str] = None,
        incercare: int = 1
) -> ReasonedComparisonResult:
    """
    PASUL 1: Generator - produce comparație cu raționament explicit și confidence score.
    Instructor garantează structura ReasonedComparisonResult.
    """
    logger.info(f"[Generator] Încercarea {incercare}/{MAX_RETRIES}")

    system_prompt, user_prompt = _build_generator_prompt(
        prod_a, prod_b, preferinte, feedback_anterior, incercare
    )

    try:
        result = instructor_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_model=ReasonedComparisonResult,
            max_retries=2,
            temperature=0,
            max_tokens=4000
        )

        logger.info(
            f"[Generator] Succes | confidence={result.confidence:.2f} | "
            f"features={len(result.features)} | winner={result.verdict.câștigător}"
        )
        return result

    except Exception as e:
        raise HTTPException(503, f"Generator error (încercarea {incercare}): {str(e)}")


# =============================================================================
# PIPELINE: VERIFICATOR
# =============================================================================

async def verificator_step(
        generated: ReasonedComparisonResult,
        preferinte: str
) -> VerificationResult:
    """
    PASUL 2: Verificator - evaluează validitatea logicii din Generator.
    Analizează și scorul de confidence al Generatorului.

    Returnează: decizie (da/nu/nesigur) + motiv + confidence ajustat.
    """
    logger.info(f"[Verificator] Evaluez răspunsul cu confidence={generated.confidence:.2f}")

    system_prompt = """Ești un verificator critic al comparațiilor de produse.

ROLUL TĂU:
Evaluezi dacă raționamentul unui Generator este valid, consistent și corect față de preferințele userului.

CRITERII DE EVALUARE:
1. LOGICĂ: Raționamentul din 'gandire' susține verdictul?
2. CONSISTENȚĂ: Câștigătorul din verdict este consistent cu scorurile din features?
3. RELEVANȚĂ: Feature-urile comparate sunt relevante pentru preferințele userului?
4. ACURATEȚE: Nu există afirmații inventate sau contradictorii?
5. CONFIDENCE: Scorul de confidence al Generatorului este justificat față de calitatea datelor?

DECIZIE:
- 'da': Logică validă, confidence justificat → acceptă
- 'nu': Probleme grave → respinge cu feedback specific
- 'nesigur': Probleme minore sau ambiguități → retrimite cu clarificări

IMPORTANT: Dacă respingi ('nu'), câmpul 'feedback_pentru_retry' trebuie să fie specific și acționabil."""

    user_prompt = f"""Verifică această comparație pentru userul care vrea: "{preferinte}"

--- RAȚIONAMENT GENERATOR ---
GÂNDIRE: {generated.gandire}

CONFIDENCE GENERATOR: {generated.confidence:.2f}
JUSTIFICARE CONFIDENCE: {generated.confidence_rationale}

--- REZULTAT ---
Produs A: {generated.produs_a_titlu}
Produs B: {generated.produs_b_titlu}
Câștigător: {generated.verdict.câștigător}
Scor A: {generated.verdict.scor_a} | Scor B: {generated.verdict.scor_b}
Diferență semnificativă: {generated.verdict.diferență_semificativă}
Argument principal: {generated.verdict.argument_principal}
Compromisuri: {generated.verdict.compromisuri}

FEATURES COMPARATE ({len(generated.features)} total):
{chr(10).join(f"- {f.feature_name}: A={f.produs_a_value} vs B={f.produs_b_value} → Winner={f.winner} (score={f.winner_score})" for f in generated.features)}

Preferinte procesate: {generated.preferinte_procesate}
---

Evaluează:
1. Logica din GÂNDIRE susține verdictul?
2. Scorurile features sunt consistente cu câștigătorul final?
3. Confidence-ul de {generated.confidence:.2f} este justificat?
4. Există probleme de acuratețe sau relevanță față de preferințele userului?"""

    try:
        result = instructor_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_model=VerificationResult,
            max_retries=2,
            temperature=0,
            max_tokens=1500
        )

        logger.info(
            f"[Verificator] Decizie={result.decizie} | "
            f"confidence_ajustat={result.confidence_adjusted:.2f} | "
            f"probleme={len(result.probleme_identificate)}"
        )
        return result

    except Exception as e:
        raise HTTPException(503, f"Verificator error: {str(e)}")


# =============================================================================
# PIPELINE ORCHESTRATOR
# =============================================================================

async def pipeline_compara(
        prod_a: ProductData,
        prod_b: ProductData,
        preferinte: str
) -> FinalComparisonResult:
    """
    Orchestrează pipeline-ul complet:
    Generator → Verificator → [Retry dacă respins] → Rezultat final

    Max MAX_RETRIES încercări. Dacă după 3 încercări tot e respins,
    returnează cel mai bun rezultat disponibil cu warning.
    """
    pipeline_log: List[str] = []
    best_result: Optional[ReasonedComparisonResult] = None
    best_verification: Optional[VerificationResult] = None
    feedback_anterior: Optional[str] = None

    for incercare in range(1, MAX_RETRIES + 1):
        log_prefix = f"[Încercarea {incercare}/{MAX_RETRIES}]"
        pipeline_log.append(f"{log_prefix} Generator pornit...")

        # --- PASUL 1: GENERATOR ---
        generated = await generator_step(
            prod_a, prod_b, preferinte,
            feedback_anterior=feedback_anterior,
            incercare=incercare
        )
        pipeline_log.append(
            f"{log_prefix} Generator finalizat | "
            f"confidence={generated.confidence:.2f} | "
            f"winner={generated.verdict.câștigător}"
        )

        # --- PASUL 2: VERIFICATOR ---
        pipeline_log.append(f"{log_prefix} Verificator pornit...")
        verification = await verificator_step(generated, preferinte)
        pipeline_log.append(
            f"{log_prefix} Verificator decizie={verification.decizie} | "
            f"confidence_ajustat={verification.confidence_adjusted:.2f}"
        )

        if verification.probleme_identificate:
            pipeline_log.append(
                f"{log_prefix} Probleme identificate: "
                + " | ".join(verification.probleme_identificate)
            )

        # Salvăm cel mai bun rezultat (preferăm 'da', apoi 'nesigur', apoi orice)
        if best_result is None or verification.decizie == "da":
            best_result = generated
            best_verification = verification

        # --- DECIZIE RETRY ---
        if verification.decizie == "da":
            pipeline_log.append(f"{log_prefix} ✅ Verificare ACCEPTATĂ. Pipeline complet.")
            break

        elif verification.decizie == "nu":
            if incercare < MAX_RETRIES:
                feedback_anterior = verification.feedback_pentru_retry or verification.motiv
                pipeline_log.append(
                    f"{log_prefix} ❌ Verificare RESPINSĂ. "
                    f"Retry cu feedback: {feedback_anterior[:100]}..."
                )
            else:
                pipeline_log.append(
                    f"{log_prefix} ❌ Verificare RESPINSĂ. "
                    f"Număr maxim de reîncercări atins. Se returnează cel mai bun rezultat."
                )

        elif verification.decizie == "nesigur":
            if incercare < MAX_RETRIES:
                feedback_anterior = verification.feedback_pentru_retry or verification.motiv
                pipeline_log.append(
                    f"{log_prefix} ⚠️ Verificare NESIGURĂ. "
                    f"Retry cu clarificări: {feedback_anterior[:100]}..."
                )
            else:
                pipeline_log.append(
                    f"{log_prefix} ⚠️ Verificare NESIGURĂ. Max reîncercări atins."
                )
                break

    # --- CONSTRUIM REZULTATUL FINAL ---
    return FinalComparisonResult(
        # Rezultatul comparației
        produs_a_titlu=best_result.produs_a_titlu,
        produs_b_titlu=best_result.produs_b_titlu,
        features=best_result.features,
        verdict=best_result.verdict,
        preferinte_procesate=best_result.preferinte_procesate,

        # Metadata pipeline
        gandire=best_result.gandire,
        confidence_generator=best_result.confidence,
        confidence_verificator=best_verification.confidence_adjusted,
        confidence_rationale=best_result.confidence_rationale,
        verificare_decizie=best_verification.decizie,
        numar_incercari=incercare,
        pipeline_log=pipeline_log,
    )


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="Product Comparison cu Instructor + Pipeline Verificare",
    description="""
Comparare produse cu raționament explicit și validare automată via pipeline în 2 pași.

**Pipeline:**
1. **Generator**: LLM produce comparație cu GÂNDIRE explicită + RĂSPUNS + confidence score
2. **Verificator**: Al doilea LLM evaluează validitatea logicii (da/nu/nesigur) și ajustează confidence-ul
3. **Retry**: Dacă respins, se retrimite cu feedback specific (max 3 încercări)

**Câmpuri cheie în răspuns:**
- `gandire`: Raționamentul explicit al Generatorului
- `confidence_generator`: Cât de sigur a fost Generatorul (0.0-1.0)
- `confidence_verificator`: Confidence ajustat după verificare
- `verificare_decizie`: Decizia Verificatorului (da/nu/nesigur)
- `numar_incercari`: Câte încercări au fost necesare
- `pipeline_log`: Log detaliat al pașilor
""",
    version="4.0.0"
)


@app.post("/compare", response_model=FinalComparisonResult)
async def compare(request: ComparisonRequest):
    """
    Compară două produse cu pipeline Generator → Verificator → Retry.

    **Exemplu cu text:**
    ```json
    {
        "produs_a": {"sursa": "MacBook Air M3 8GB 256GB 1.24kg", "este_url": false},
        "produs_b": {"sursa": "ThinkPad X1 i7 16GB 512GB 1.13kg", "este_url": false},
        "preferinte": "Dezvoltare software și transport zilnic"
    }
    ```
    """
    import time
    start = time.time()

    # Extrage date produse
    if request.produs_a.este_url:
        date_a = await scrape_product(request.produs_a.sursa)
    else:
        date_a = parse_text_input(request.produs_a.sursa)

    if request.produs_b.este_url:
        date_b = await scrape_product(request.produs_b.sursa)
    else:
        date_b = parse_text_input(request.produs_b.sursa)

    # Pipeline complet
    result = await pipeline_compara(date_a, date_b, request.preferinte)

    elapsed_ms = int((time.time() - start) * 1000)
    logger.info(
        f"[API] Finalizat în {elapsed_ms}ms | "
        f"încercări={result.numar_incercari} | "
        f"decizie={result.verificare_decizie} | "
        f"confidence_final={result.confidence_verificator:.2f}"
    )

    return result


@app.get("/health")
async def health():
    """Verificare stare."""
    try:
        client.models.list()
        api_ok = True
    except Exception:
        api_ok = False

    return {
        "status": "ok" if api_ok else "degraded",
        "instructor": "active",
        "model": MODEL,
        "mode": "instructor-json",
        "pipeline": {
            "generator": "ReasonedComparisonResult",
            "verificator": "VerificationResult",
            "max_retries": MAX_RETRIES,
            "confidence_tracking": True
        }
    }


@app.delete("/cache")
async def clear_cache():
    """Golește cache."""
    cache.clear()
    return {"message": "Cache cleared"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)