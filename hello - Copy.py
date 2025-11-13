import re
from typing import Dict, List, Any, Optional
from config import Config
import traceback

# Defer imports for conditional loading

class LLMService:
    """Service for LLM-based code review - supports OpenAI, Google Gemini, and OpenRouter"""

    def __init__(self):
        self.config = Config()
        self.provider = self.config.LLM_PROVIDER

        self.model_name: str = ""
        # The client variable will hold the OpenAI SDK instance, used for both OpenAI and OpenRouter
        self.client = None
        self.model = None  # For Gemini

        if self.provider == "gemini":
            self._init_gemini()
        elif self.provider == "openai":
            self._init_openai()
        elif self.provider == "openrouter": # <-- NEW OPENROUTER CHECK
            self._init_openrouter()
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    def _init_gemini(self) -> None:
        """Initialize Google Gemini (FREE)"""
        try:
            import google.generativeai as genai

            if not self.config.GEMINI_API_KEY:
                print("⚠️  Warning: GEMINI_API_KEY not set. Cannot use Gemini.")
                raise EnvironmentError("GEMINI_API_KEY not set.")

            genai.configure(api_key=self.config.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(self.config.GEMINI_MODEL)
            self.model_name = self.config.GEMINI_MODEL
            print(f"✅ LLM Service initialized with Google Gemini: {self.model_name}")
        except Exception as e:
            print(f"❌ Gemini initialization error: {e}")
            raise

    def _init_openai(self) -> None:
        """Initialize OpenAI"""
        try:
            from openai import OpenAI

            if not self.config.OPENAI_API_KEY:
                print("⚠️  Warning: OPENAI_API_KEY not set. Cannot use OpenAI.")
                raise EnvironmentError("OPENAI_API_KEY not set.")

            # OpenAI client uses default base URL
            self.client = OpenAI(api_key=self.config.OPENAI_API_KEY)
            self.model_name = self.config.OPENAI_MODEL
            print(f"✅ LLM Service initialized with OpenAI: {self.model_name}")
        except Exception as e:
            print(f"❌ OpenAI initialization error: {e}")
            raise

    def _init_openrouter(self) -> None: # <-- NEW METHOD
        """Initialize OpenRouter using the OpenAI SDK's compatibility mode"""
        try:
            from openai import OpenAI
            
            if not self.config.OPENROUTER_API_KEY:
                print("⚠️  Warning: OPENROUTER_API_KEY not set.")
                raise EnvironmentError("OPENROUTER_API_KEY not set.")

            # OpenRouter uses the OpenAI SDK but requires the base_url to be set
            self.client = OpenAI(
                api_key=self.config.OPENROUTER_API_KEY,
                base_url=self.config.OPENROUTER_BASE_URL
            )
            self.model_name = self.config.OPENROUTER_MODEL
            print(f"✅ LLM Service initialized with OpenRouter: {self.model_name}")
        except Exception as e:
            print(f"❌ OpenRouter initialization error: {e}")
            raise

    def generate_code_review(
        self,
        code_content: str,
        static_analysis: Dict[str, Any],
        relevant_context: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Generate comprehensive code review"""
        try:
            prompt = self._build_review_prompt(
                code_content, static_analysis, relevant_context or []
            )

            print(f"🤖 Requesting code review from {self.provider.upper()}...")

            # Call appropriate provider
            if self.provider == "gemini":
                review_text = self._call_gemini(prompt)
            elif self.provider in ["openai", "openrouter"]: # <-- Updated to include openrouter
                review_text = self._call_openai_chat(prompt)
            else:
                raise ValueError(f"Invalid LLM provider selected: {self.provider}")

            print("✅ Code review generated successfully")

            return {
                "review": review_text,
                "suggestions": self._extract_suggestions(review_text),
                "security_concerns": self._extract_security_concerns(review_text),
                "refactoring_ideas": self._extract_refactoring_ideas(review_text),
                "model": self.model_name,
            }

        except Exception as e:
            print(f"❌ Error in LLM review ({self.provider}): {e}")
            traceback.print_exc()
            return {
                "review": f"Error generating AI review from {self.provider.upper()}: {str(e)}",
                "suggestions": ["Unable to generate suggestions due to API error."],
                "security_concerns": [],
                "refactoring_ideas": [],
                "model": self.model_name,
            }

    # --- NEW METHOD FOR CODE REFACTORING ---
    def get_updated_code(self, code_content: str, prompt: str) -> str:
        """
        Generates and returns the refactored, updated code from the LLM.
        """
        full_prompt = f"{prompt}\n\nCode to refactor:\n\n```python\n{code_content}\n```"

        try:
            # Call appropriate provider
            if self.provider == "gemini":
                response_text = self._call_gemini(full_prompt)
            elif self.provider in ["openai", "openrouter"]: # <-- Updated to include openrouter
                response_text = self._call_openai_chat(full_prompt)
            else:
                raise ValueError(f"Invalid LLM provider selected: {self.provider}")

            # Clean up the response: remove markdown fences (```python...```) 
            cleaned_code = re.sub(r'```python\n|```', '', response_text, flags=re.DOTALL).strip()
            
            return cleaned_code
        
        except Exception as e:
            print(f"❌ LLM code update failed ({self.provider}): {e}")
            traceback.print_exc()
            raise 

    # =========================================================================
    # Gemini Call Method 
    # =========================================================================
    def _call_gemini(self, prompt: str) -> str:
        """Call Google Gemini model to generate review text"""
        try:
            import google.generativeai as genai

            safety_settings = {
                genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            }

            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=3000,
                    top_p=0.95,
                    top_k=40,
                ),
                safety_settings=safety_settings,
            )

            # --- CHECK FINISH REASON BEFORE ACCESSING TEXT ---
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                finish_reason = candidate.finish_reason
                                
                if finish_reason == 2:  # SAFETY block
                    print("⚠️ Gemini blocked response due to safety filters")
                    raise ValueError("Content blocked by Gemini safety filters. Try uploading a different file or use simpler code.")
                elif finish_reason == 3:  # RECITATION (copyright)
                    print("⚠️ Gemini blocked response due to recitation/copyright")
                    raise ValueError("Content blocked due to potential copyright concerns.")

            # Check if response has content
            if not getattr(response, "parts", None) and not getattr(response, "text", None):
                feedback = getattr(response, "prompt_feedback", None)
                if feedback:
                    print(f"⚠️ Gemini blocked response. Reason: {feedback}")
                    raise ValueError(f"Content blocked by Gemini: {feedback}")
                
                raise ValueError("Empty response from Gemini. The prompt may have triggered an internal policy block.") 

            # Get text from response
            if hasattr(response, "text") and response.text:
                return response.text
            else:
                parts = [p.text for p in getattr(response, "parts", []) if hasattr(p, "text")]
                return "\n".join(parts) if parts else ""

        except ValueError as e:
            print(f"❌ Gemini content blocked: {e}")
            raise
        except Exception as e:
            print(f"❌ Gemini API error: {e}")
            raise
    # =========================================================================


    def _call_openai_chat(self, prompt: str) -> str:
        """
        Call OpenAI or OpenRouter API. 
        Note: self.client is configured with the correct base_url based on the provider.
        """
        try:
            from openai import OpenAI
            
            # The OpenRouter model name is stored in self.model_name
            # The OpenRouter base_url is set during initialization in self.client
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert code reviewer with deep knowledge of software engineering best practices, security, and code quality. When asked to fix code, you must output ONLY the complete, corrected code.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=3000, # Increased tokens for full code generation
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ {self.provider.capitalize()} API error: {e}")
            raise

    def _build_review_prompt(
        self, code: str, static_analysis: Dict[str, Any], relevant_context: List[Dict[str, Any]]
    ) -> str:
        """Build comprehensive prompt for code review"""

        issues = static_analysis.get("issues", [])[:5]
        issues_text = "\n".join(
            [
                f"- Line {issue.get('line', 'N/A')}: {issue.get('message', 'No message')} ({issue.get('severity', 'Info')})"
                for issue in issues
            ]
        )

        prompt = f"""Review this Python code and provide comprehensive feedback:

## Code to Review:
{code[:3000]}

## Static Analysis Issues Found:
{issues_text if issues else "No major issues detected"}

"""
        # Add RAG context if available
        if relevant_context:
            prompt += "\n## Similar Code Patterns Found (RAG Context):\n"
            for i, ctx in enumerate(relevant_context[:3], 1):
                prompt += f"\n{i}. {ctx.get('name', 'Unknown')} (similarity: {ctx.get('similarity', 0):.2f})\n"
                prompt += f"  Context Snippet: {ctx.get('text', '')[:200]}...\n"

        prompt += """
## Please provide your review in the following sections:
1. **Overall Assessment**: Brief quality summary (2-3 sentences)
2. **Critical Issues**: Security, bugs, performance problems
3. **Code Quality**: Best practices, readability, structure
4. **Refactoring Suggestions**: Specific improvements
5. **Positive Aspects**: What's done well

Be specific and actionable. Use bullet points within sections 2, 3, 4, and 5."""
        return prompt

    def _extract_suggestions(self, review: str) -> List[str]:
        """Extract actionable suggestions by looking for bullet points after section titles"""
        suggestions: List[str] = []

        # Look in Critical Issues (section 2)
        crit_issues = self._extract_section_content(review, "Critical Issues")
        if crit_issues:
            suggestions.extend(crit_issues)

        # Look in Code Quality (section 3)
        code_quality = self._extract_section_content(review, "Code Quality")
        if code_quality:
            suggestions.extend(code_quality)

        return suggestions[:10]

    def _extract_security_concerns(self, review: str) -> List[str]:
        """Extract security concerns explicitly from Critical Issues section"""
        security_keywords = ["security", "vulnerability", "injection", "xss", "csrf", "secret", "key", "auth"]
        concerns: List[str] = []

        critical_issues = self._extract_section_content(review, "Critical Issues")

        for issue in critical_issues:
            if any(keyword in issue.lower() for keyword in security_keywords):
                concerns.append(issue)

        return concerns[:5]

    def _extract_refactoring_ideas(self, review: str) -> List[str]:
        """Extract refactoring ideas from the Refactoring Suggestions section"""
        return self._extract_section_content(review, "Refactoring Suggestions")[:5]

    def _extract_section_content(self, review_text: str, section_title: str) -> List[str]:
        """
        Extract bullet points from a specific section of the review text using multiple patterns.
        """
        if not review_text:
            return []

        escaped_title = re.escape(section_title)
        section_content: Optional[str] = None

        # Multiple patterns to match different section heading formats from LLMs
        patterns = [
            # Pattern 1: ## Section Title (or ###, #) followed by optional colon
            rf'#+\s*{escaped_title}\s*:?\s*\n(.*?)(?=\n#+\s*|\Z)',
            # Pattern 2: X. **Section Title** (handles the numbered, bolded list)
            rf'\d+\.\s*\*\*{escaped_title}\*\*\s*:?\s*\n(.*?)(?=\n\d+\.\s*\*\*.*?\*\*|\Z)',
            # Pattern 3: **Section Title** (handles just bolded, unnumbered)
            rf'\*\*{escaped_title}\*\*\s*:?\s*\n(.*?)(?=\n\*\*.*?\*\*|\Z)',
            # Pattern 4: Plain text "Section Title:" (Simple fallback)
            rf'{escaped_title}\s*:?\s*\n(.*?)(?=\n[A-Z][a-z]+.*?:|\Z)',
        ]

        # Try each pattern
        for pattern in patterns:
            # DOTALL ensures '.' matches newlines
            match = re.search(pattern, review_text, re.DOTALL | re.IGNORECASE)
            if match:
                # Group 1 contains the content after the title
                section_content = match.group(1).strip()
                break

        if not section_content:
            return []

        # --- Extract bullet points from the captured content ---
        suggestions: List[str] = []
        lines = section_content.split("\n")

        for line in lines:
            line = line.strip()
            # Look for standard bullet points (*, -, •) or numbered items (1., 2.)
            # The regex handles the prefix being at the start of the line or potentially indented
            if re.match(r'^[-*•]|^(\d+\.)', line):
                # Clean up the prefix
                suggestion = re.sub(r'^[-*•]|^(\d+\.)\s*', '', line).strip()
                if len(suggestion) > 10: 
                    suggestions.append(suggestion)
        return suggestions