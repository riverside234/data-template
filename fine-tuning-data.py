import os
import re
import json
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from dotenv import load_dotenv

try:
    import openai
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except Exception:
    HAS_ANTHROPIC = False

load_dotenv()

# ---------------- config ----------------
SEED = int(os.getenv("SEED", "17"))
random.seed(SEED)

DOMAINS = [
    "Role-Play and Simulation – characters, worlds, scenarios",
    "Marketing, Brand and Social Strategy – positioning, campaigns, community",
    "Entertainment, Media and Pop Culture – industry trends, fandoms, critique",
    "Politics and Public Policy – domestic governance, policy analysis",
    "Sports and Athletics – rules, tactics, commentary",
    "Education and Instructional Design – curricula, pedagogy, assessment",
    "History and Historiography – events, sources, methods",
    "Science Communication – explainers for lay audiences",
    "Philosophy and Ethics – logic, moral frameworks, dilemmas",
    "Psychology and Behavioral Science – cognition, behavior, motivation",
    "Business Strategy and Leadership – org models, scaling, executive practice",
    "Career Coaching and Job Search – resumes, interviews, growth plans",
    "Health and Medicine – conditions, care models, health systems",
    "Software and Digital Transformation – development topics, IT modernization",
    "Environment and Sustainability – climate, conservation, stewardship",
    "Economics and Markets – theory, policy, market dynamics",
    "Law and Legal Literacy – rights, procedures, legal concepts",
    "Human Resources and People Ops – hiring, performance, culture",
    "Project and Program Management – planning, delivery, PMO",
    "Corporate Finance and Investing – valuation, portfolio theory",
    "Real Estate and Urban Development – property, zoning, city growth",
    "Travel and Tourism – itineraries, cultural etiquette, destinations",
    "Food and Culinary Arts – techniques, cuisines, menu ideas",
    "Fashion and Personal Style – aesthetics, wardrobe systems",
    "Creative Arts and Music Production – visual design, composition, recording",
    "Gaming and Interactive Media – mechanics, design, esports",
    "Relationships, Family and Parenting – communication, development, dynamics",
    "Personal Productivity and Time Management – workflows, habits, tools",
    "Home, DIY and Gardening – projects, materials, horticulture",
    "Pet Care and Animal Behavior – training, enrichment, health basics",
    "Automotive and Mobility – vehicles, maintenance, transport tech",
    "Fitness and Nutrition – training programs, diet planning, performance",
    "Mental Health and Wellbeing – coping skills, mindfulness, supports",
    "Spirituality and Comparative Religion – beliefs, practices, traditions",
    "Social and Cultural Studies – institutions, norms, anthropology",
    "International Relations and Geopolitics – states, alliances, strategy",
    "Journalism and Media Literacy – reporting, verification, bias detection",
    "Communication, Persuasion and Rhetoric – argumentation, framing, style",
    "Artificial Intelligence and Data Science – models, analytics, insights",
    "Innovation and Design Thinking – discovery, prototyping, iteration",
    "Entrepreneurship and Startups – idea validation, traction, funding",
]


DIALOGUES_PER_DOMAIN = int(os.getenv("DIALOGUES_PER_DOMAIN", "300"))

# Teacher weights 
TEACHER_WEIGHTS = {
    #(provider, model, weight)
    "grok-mini": ("xai", "grok-3-mini", float(os.getenv("W_GROK", "0.67"))),
    "gpt4o-mini": ("openai", "gpt-4o-mini", float(os.getenv("W_GPT4OMINI", "0.15"))),
    "claude-haiku": ("anthropic", "claude-3-haiku-20240307", float(os.getenv("W_HAIKU", "0.18"))),
}

# Quality filter, score
FILTER_MODEL_PREF = os.getenv("FILTER_MODEL_PREF", "grok-mini")   
FILTER_MIN_SCORE = int(os.getenv("FILTER_MIN_SCORE", "7"))
REGEN_MAX_TRIES = int(os.getenv("REGEN_MAX_TRIES", "2"))

# Output  
OUT_RAW = Path("dataset/raw/dialogues.jsonl")
OUT_TRAIN_MSG = Path("dataset/llama3_chat/train.messages.jsonl")
OUT_TEST_MSG = Path("dataset/llama3_chat/test.messages.jsonl")
for p in [OUT_RAW.parent, OUT_TRAIN_MSG.parent, OUT_TEST_MSG.parent]:
    p.mkdir(parents=True, exist_ok=True)

#----------------------------------------------------
class LLMRouter:
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.has_openai = HAS_OPENAI and bool(self.openai_key)

        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self.has_anthropic = HAS_ANTHROPIC and bool(self.anthropic_key)

        self.xai_key = os.getenv("XAI_API_KEY")
        self.xai_base_url = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
        self.has_xai = bool(self.xai_key)

        self._openai_client = None
        self._anthropic_client = None
        self._xai_client = None

        # normalize weights  
        self.teachers = []
        total = 0.0
        for tid, (prov, model, w) in TEACHER_WEIGHTS.items():
            if prov == "openai" and self.has_openai and w > 0:
                self.teachers.append((tid, prov, model, w))
                total += w
            elif prov == "anthropic" and self.has_anthropic and w > 0:
                self.teachers.append((tid, prov, model, w))
                total += w
            elif prov == "xai" and self.has_xai and w > 0:
                self.teachers.append((tid, prov, model, w))
                total += w
        if not self.teachers:
            raise RuntimeError("No LLM providers available.")
        self.teachers = [(tid, prov, model, w/total) for (tid, prov, model, w) in self.teachers]

    def chat_json(self, teacher_id: Optional[str], system: str, user: str, temperature=0.6, max_tokens=2000) -> dict:
        tid, prov, model = self._pick_teacher(teacher_id)
        content = self._chat_call(prov, model, system, user, temperature, max_tokens, expect_json=True)
        return self._force_json(content)

    def chat_text(self, teacher_id: Optional[str], system: str, user: str, temperature=0.8, max_tokens=2000) -> str:
        tid, prov, model = self._pick_teacher(teacher_id)
        return self._chat_call(prov, model, system, user, temperature, max_tokens, expect_json=False)

    def _pick_teacher(self, teacher_id: Optional[str]) -> Tuple[str, str, str]:
        if teacher_id:
            for tid, prov, model, _ in self.teachers:
                if tid == teacher_id:
                    return tid, prov, model
        r = random.random()
        acc = 0.0
        for tid, prov, model, w in self.teachers:
            acc += w
            if r <= acc:
                return tid, prov, model
        return self.teachers[-1][0:3]

    def _oa_client(self):
        if not self._openai_client:
            if not self.has_openai:
                raise RuntimeError("OPENAI_API_KEY is not set.")
            self._openai_client = openai.OpenAI(api_key=self.openai_key)
        return self._openai_client

    def _anth_client(self):
        if not self._anthropic_client:
            if not self.has_anthropic:
                raise RuntimeError("ANTHROPIC_API_KEY is not set.")
            self._anthropic_client = anthropic.Anthropic(api_key=self.anthropic_key)
        return self._anthropic_client

    def _xai_client(self):
        if not self._xai_client:
            if not self.has_xai:
                raise RuntimeError("XAI_API_KEY is not set.")
            self._xai_client = openai.OpenAI(api_key=self.xai_key, base_url=self.xai_base_url)
        return self._xai_client

    def _chat_call(self, prov: str, model: str, system: str, user: str, temperature: float, max_tokens: int, expect_json: bool) -> str:
        if prov == "openai":
            client = self._oa_client()
            params = {
                "model": model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if expect_json:
                params["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(**params)
            return resp.choices[0].message.content

        elif prov == "anthropic":
            client = self._anth_client()
            msg = client.messages.create(
                model=model,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": user}],
            )
            content = "".join([c.text for c in msg.content if hasattr(c, "text")])
            return content

        elif prov == "xai":
            client = self._xai_client()
            params = {
                "model": model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if expect_json:
                params["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(**params)
            return resp.choices[0].message.content
        else:
            raise ValueError(f"Unknown provider: {prov}")

    @staticmethod
    def _force_json(s: str) -> dict:
        try:
            return json.loads(s)
        except Exception:
            pass
        try:
            s2 = re.search(r"\{.*\}", s, re.DOTALL)
            if s2:
                return json.loads(s2.group(0))
        except Exception:
            pass
        raise ValueError("Model did not return valid JSON:\n" + s)


router = LLMRouter()

# ---------------- data generation----------------

def gen_user_intent_and_prefs(domain: str, teacher_hint: Optional[str] = None) -> str:
    """
    turn1 
    """
    style = random.choices(["detailed", "underspecified"], weights=[0.55, 0.45])[0]
    system = (
        "You write realistic end-user intents for an LLM in the given domain.\n"
        "Produce ONE paragraph that begins directly with the user's intent (no labels, no quotes).\n"
        "Quality requirements:\n"
        "• Vary specificity: if Style=detailed, write 3–5 sentences with concrete details; if Style=underspecified, write 1–2 sentences with broad goals.\n"
        "• Integrate at least TWO explicit user preferences into the prose (tone, length, style, constraints, persona, sources, formatting, safety).\n"
        "• Natural language, first-person or imperative voice; avoid meta-text about prompts/LLMs.\n"
        "• 50–120 words; no lists, no placeholders like [X], no fenced code, no URLs.\n"
        "• Keep content lawful and safe; avoid disallowed or sensitive topics."
    )
    user = f"Domain: {domain}\nStyle: {style}\nInstruction: Write only the paragraph."
    return router.chat_text(teacher_hint, system, user, temperature=0.9, max_tokens=240)


def turn2_intent_understanding_and_base_prompt(domain: str, intent_text: str, teacher_hint: Optional[str] = None) -> dict:
    """
    turn2 
    """
    system = (
        "You are designing a base agent for the provided user intent. Extract intent, analyze outcome, collect capabilities, "
        "then propose a brief plan and a simple seed prompt.\n"
        "Return a MINIFIED JSON object with EXACT keys and types:\n"
        "{"
        "\"purpose\": str, "
        "\"context\": str, "
        "\"desired_outcome\": str, "
        "\"capability_information\": {"
        "\"explicit_inferred_capabilities\": [str,...], "
        "\"task_required_capabilities\": [str,...]"
        "}, "
        "\"agent_plan\": str, "
        "\"initial_prompt\": str"
        "}\n"
        "Constraints:\n"
        "• 'explicit_inferred_capabilities' must be capabilities explicitly mentioned or clearly implied by the user's text.\n"
        "• 'task_required_capabilities' are generic abilities an LLM agent would need for this task type.\n"
        "• Each list contains 5–9 short, non-overlapping items (deduplicate; order by importance).\n"
        "• 'agent_plan' is 3–6 concise sentences that describe the agent's steps.\n"
        "• 'initial_prompt' is a compact, human-readable instruction (<=120 words) suitable for an LLM, starting with \"You are ...\".\n"
        "• No additional fields, no markdown fences, no trailing commas, strictly valid JSON."
    )
    user = f"Domain: {domain}\nUser intent and preferences:\n{intent_text}"
    return router.chat_json(None, system, user, temperature=0.5, max_tokens=1300)


def turn3_prompt_optimization_suggestions(domain: str, turn2: dict, teacher_hint: Optional[str] = None) -> dict:
    """
    turn3
    """
    system = (
        "You optimize the agent's prompt while preserving the user's intent.\n"
        "Return a MINIFIED JSON object with EXACT keys:\n"
        "{"
        "\"summary\": {\"purpose\": str, \"context\": str, \"desired_outcome\": str}, "
        "\"optimized_capabilities\": [str,...], "
        "\"plan_prompt_improvement\": str, "
        "\"optimization_suggestions\": ["
        "{\"suggestion_number\": 1, \"title\": str, \"description\": str}, "
        "{\"suggestion_number\": 2, \"title\": str, \"description\": str}, "
        "{\"suggestion_number\": 3, \"title\": str, \"description\": str}, "
        "{\"suggestion_number\": 4, \"title\": str, \"description\": str}, "
        "{\"suggestion_number\": 5, \"title\": str, \"description\": str}"
        "]"
        "}\n"
        "Requirements:\n"
        "• 'optimized_capabilities' rebalances and merges the two sources from Turn-2; provide 6–10 deduplicated items in priority order.\n"
        "• The five suggestions MUST correspond exactly to: (1) define an appropriate agent role; "
        "(2) use precise, domain-specific terminology; (3) provide necessary extra background context; "
        "(4) add missing details for completeness; (5) specify output format, length, and formatting.\n"
        "• Each 'description' is 1–3 sentences; concrete and actionable; do NOT write the final prompt here.\n"
        "• Strict JSON only; no markdown, no extra keys."
    )
    user = "Use this Turn-2 artifact as input:\n" + json.dumps(turn2, ensure_ascii=False)
    return router.chat_json(None, system, user, temperature=0.6, max_tokens=1300)


def turn4_final_prompt(domain: str, turn2: dict, turn3: dict, teacher_hint: Optional[str] = None) -> dict:
    """
    turn4: generate the final optimized prompt  
    """
    system = (
        "Write the FINAL optimized prompt for the agent, guided by the plan and the five suggestions.\n"
        "Return STRICT JSON with exactly one key: {\"optimized_prompt\": str}.\n"
        "Prompt requirements:\n"
        "• Start with a clear role line (e.g., \"You are a ...\").\n"
        "• Include structured sections with headings such as: Role, Objectives, Context, Capabilities to Apply (use 'optimized_capabilities'), "
        "Plan/Process, Constraints, and Output Requirements (format, length, tone).\n"
        "• The prompt must be self-contained, concrete, and immediately usable by an LLM. No meta-commentary.\n"
        "• 180–350 words; use crisp, domain-appropriate terminology; avoid hallucinating facts.\n"
        "• Strict JSON output; no code fences, no extra keys."
    )
    user = (
        f"Domain: {domain}\n"
        "Turn-2:\n" + json.dumps(turn2, ensure_ascii=False) + "\n\n"
        "Turn-3:\n" + json.dumps(turn3, ensure_ascii=False)
    )
    return router.chat_json(None, system, user, temperature=0.6, max_tokens=1300)


def judge_quality(domain: str, intent_text: str, t2: dict, t3: dict, t4: dict) -> dict:
    """
    Quality filter with a transparent rubric.
    Returns: {'keep': bool, 'score': int (0-10), 'reason': str}
    """
    judge_tid = FILTER_MODEL_PREF if any(tid == FILTER_MODEL_PREF for tid, *_ in router.teachers) else None
    system = (
        "You are a strict QA judge for prompt-synthesis dialogues.\n"
        "Score the FINAL optimized prompt using this rubric (0–10 total):\n"
        "  (a) Schema compliance (Turn-2 & Turn-3 fields present; Turn-4 JSON correct) — 0–2\n"
        "  (b) Alignment with Turn-1 intent & preferences — 0–3\n"
        "  (c) Coverage of the FIVE required suggestions in Turn-4 content — 0–2\n"
        "  (d) Clarity, structure, and actionability of the prompt — 0–2\n"
        "  (e) Safety & specificity (no hallucinated facts; concrete output requirements) — 0–1\n"
        "Return ONLY a JSON object: {\"keep\": true|false, \"score\": int, \"reason\": str}.\n"
        f"Set keep=true only if score >= {FILTER_MIN_SCORE} AND the prompt includes role, domain terminology, background, missing details, and output spec."
    )
    user = (
        f"Domain: {domain}\n"
        "Turn-1 (User intent & preference):\n" + intent_text + "\n\n"
        "Turn-2:\n" + json.dumps(t2, ensure_ascii=False) + "\n\n"
        "Turn-3:\n" + json.dumps(t3, ensure_ascii=False) + "\n\n"
        "Turn-4:\n" + json.dumps(t4, ensure_ascii=False)
    )
    return router.chat_json(judge_tid, system, user, temperature=0.0, max_tokens=500)


# ---------------- format chat ----------------

def messages_from_turns(intent_text: str, t2: dict, t3: dict, t4: dict) -> List[dict]:

    return [
        {"role": "user", "content": intent_text},
        {"role": "assistant", "content": json.dumps(t2, ensure_ascii=False)},
        {"role": "user", "content": "Generate Turn-3 (Prompt Optimization Suggestions) as previously specified."},
        {"role": "assistant", "content": json.dumps(t3, ensure_ascii=False)},
        {"role": "user", "content": "Generate Turn-4 (Final Optimized Prompt) as previously specified."},
        {"role": "assistant", "content": json.dumps(t4, ensure_ascii=False)},
    ]


def append_jsonl(path: Path, rows: List):
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")




def main():
    raw_rows_buffer = []
    per_domain_messages: Dict[str, List[List[dict]]] = {d: [] for d in DOMAINS}

    for domain in DOMAINS:
        print(f"\n=== Domain: {domain} ===")
        for i in range(DIALOGUES_PER_DOMAIN):
            try:
                # Turns
                t1 = gen_user_intent_and_prefs(domain)

                t2 = turn2_intent_understanding_and_base_prompt(domain, t1)

                t3 = turn3_prompt_optimization_suggestions(domain, t2)
                
                #judge
                regen_tries = 0
                while True:
                    t4 = turn4_final_prompt(domain, t2, t3)
                    verdict = judge_quality(domain, t1, t2, t3, t4)
                    if verdict.get("keep") and int(verdict.get("score", 0)) >= FILTER_MIN_SCORE:
                        break
                    regen_tries += 1
                    if regen_tries > REGEN_MAX_TRIES:
                        break
                    time.sleep(0.2)

                # record
                row = {
                    "id": f"{hash((domain, t1)) & 0xffffffff:x}-{i}",
                    "domain": domain,
                    "teacher_pool": [t[0] for t in router.teachers],
                    "turn1_user_intent_and_preference": t1,
                    "turn2_intent_understanding_and_base_prompt": t2,
                    "turn3_prompt_optimization_suggestions": t3,
                    "turn4_final_prompt": t4,
                    "filter_verdict": verdict,
                }
                raw_rows_buffer.append(row)

                # conversation list  
                messages = messages_from_turns(t1, t2, t3, t4)
                per_domain_messages[domain].append(messages)

                print(f"  ✓ dialogue {i+1}/{DIALOGUES_PER_DOMAIN} (score={verdict.get('score')}, keep={verdict.get('keep')})")

            except Exception as e:
                print(f"  ✗ dialogue {i+1} failed: {e}")

        if raw_rows_buffer:
            append_jsonl(OUT_RAW, raw_rows_buffer)
            raw_rows_buffer = []

    # Train/test split: per domain, 10 test dialogues
    train_msgs, test_msgs = [], []
    for domain, lists in per_domain_messages.items():
        random.shuffle(lists)
        test = lists[:10] if len(lists) >= 10 else lists[: max(1, len(lists)//5)]
        train = lists[len(test):]
        test_msgs.extend(test)
        train_msgs.extend(train)

    append_jsonl(OUT_TRAIN_MSG, train_msgs)
    append_jsonl(OUT_TEST_MSG, test_msgs)

    print(f"\nraw {OUT_RAW}\n - Train   {OUT_TRAIN_MSG}\n - Test  {OUT_TEST_MSG}")


if __name__ == "__main__":
    main()
