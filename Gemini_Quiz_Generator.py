import google.genai as genai
from google.genai import types
import json


class Gemini_Quiz_Generator:
    def __init__(self, api_key, model_name="gemini-2.0-flash", retries=3):
        self.api_key = api_key
        self.__model_name = model_name
        self.__client = genai.Client(api_key=self.api_key)
        self.retries = retries

    def generate_quiz(self, file_path):
        if self.retries == 0:
            raise Exception("Unable To Generate a Response")
        try:
            prompt = (
                    """
                    ## Core Objective
                    Generate high-quality, document-based multiple-choice questions that thoroughly assess comprehension and critical thinking for substantive, information-rich documents.
                    
                    ## Document Eligibility Criteria
                    ### Acceptable Document Types
                    * Academic textbooks and course materials
                    * Research papers and scholarly publications
                    * Technical documentation and manuals
                    * Professional training materials
                    * Comprehensive instructional guides
                    * Detailed research reports
                    * Historical manuscripts and archives

                    ### Excluded Document Types
                    * Social media posts
                    * Short-form content
                    * Incomplete or fragmentary documents
                    * Highly informal or conversational texts
                    * Documents with insufficient depth or complexity
                    * Marketing materials
                    * Raw data files without substantial contextual information

                    ## Comprehensive Question Generation Guidelines

                    ### Document Preprocessing
                    1. **Initial Document Assessment**
                    * Verify document meets eligibility criteria
                    * Confirm minimum required length and information density
                    * Assess overall structural coherence and depth of content

                    2. **Content Suitability Evaluation**
                    * Check for:
                        - Clear narrative or informational structure
                        - Sufficient complex ideas and concepts
                        - Potential for multi-level comprehension questions
                        - Absence of overly technical or domain-specific jargon that would limit accessibility

                    ### Question Design Criteria

                    #### Difficulty Levels
                    * **Easy-Level Questions (10 questions)**
                    - Focus on fundamental facts and directly stated information
                    - Require basic recall and surface-level understanding
                    - Use clear, straightforward language
                    - Answers should be explicitly present in the text

                    * **Medium-Level Questions (5 questions)**
                    - Require deeper comprehension and basic inference
                    - Involve synthesizing information from multiple sections
                    - Demand moderate critical thinking
                    - Answers should be derivable through careful text analysis

                    * **Hard-Level Questions (5 questions)**
                    - Challenge advanced comprehension and analytical skills
                    - Require complex reasoning and in-depth understanding
                    - Involve subtle inferences, contextual interpretation
                    - May connect different parts of the document in nuanced ways

                    #### Question Construction Standards
                    * **Accuracy**
                    - 100% grounded in document content
                    - No external knowledge or fabricated information
                    - Precise alignment with source material

                    * **Formatting**
                    * Each question must have:
                        - Clear, concise stem
                        - 4 plausible answer options
                        - One definitively correct answer
                        - Distractors that are:
                        1. Plausible but incorrect
                        2. Derived from document context
                        3. Similar in structure to correct answer

                    * **Cognitive Engagement**
                    - Promote active reading and critical analysis
                    - Avoid trivial or purely factual recall
                    - Encourage deeper document comprehension

                    ### Structured Output Format
                    ```json
                    {
                    "questions": [
                        {
                        "id": "Q1",
                        "level": "easy/medium/hard",
                        "text": "Question stem here?",
                        "options": [
                            {"text": "Option A", "isCorrect": false},
                            {"text": "Option B", "isCorrect": true},
                            {"text": "Option C", "isCorrect": false},
                            {"text": "Option D", "isCorrect": false}
                        ],
                        "explanation": "Brief rationale for the correct answer, referencing document"
                        }
                        // Additional 19 questions follow same structure
                    ]
                    }
                    ```

                    ### Additional Considerations
                    * Ensure balanced coverage of document content
                    * Avoid repeating similar question types
                    * Prioritize questions that test genuine understanding
                    * Include explanations to support learning

                    ## Final Verification Checklist
                    - [ ] Document meets eligibility criteria
                    - [ ] 20 total questions generated
                    - [ ] 10 easy-level questions
                    - [ ] 5 medium-level questions
                    - [ ] 5 hard-level questions
                    - [ ] All questions strictly based on document
                    - [ ] Comprehensive coverage of key content
                    - [ ] Clear, professional formatting
                    - [ ] Informative explanations provided

                    **Note:** Dynamically adapt the approach based on the specific document's complexity, structure, and subject matter, ensuring only high-quality, substantive documents are processed.""")

            with open(file_path, "rb") as f:
                pdf_data = f.read()
            # Create a file part from the bytes data.
            pdf_part = types.Part.from_bytes(data=pdf_data, mime_type="application/pdf")
            # Generate content using the prompt and the file part.
            response = self.__client.models.generate_content(
                model=self.__model_name,
                contents=[prompt, pdf_part],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.3,
                    top_p=0.7,
                    top_k=40,
                    max_output_tokens=8192
                )
            )
            return json.loads(response.text)
        
        except Exception as err:
            self.retries -= 1
            return self.generate_quiz(file_path=file_path)

