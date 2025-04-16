import google.generativeai as genai

class GeminiAI:
    def __init__(self):
        genai.configure(api_key="AIzaSyD6QpuXnUD0BOCEP4zBMZXDnn1C0CvgXzw")
        self.model = genai.GenerativeModel(model_name="gemini-2.0-flash")
        self.knowledge_base = []

    def add_to_knowledge_base(self, context_chunks):
        self.knowledge_base.extend(context_chunks)

    def add_multiple_to_knowledge_base(self, list_of_context_chunks):
        for context_chunks in list_of_context_chunks:
            self.add_to_knowledge_base(context_chunks)

    def get_answer(self, question, context_chunks):
        context = "\n".join(context_chunks)
        prompt = (
           f"""
            You are a helpful assistant designed to answer questions based on the following context:

            CONTEXT:
            {context}  (Content from your VectorDB)

            INSTRUCTIONS:
            1.  Answer the user's question only using information found within the CONTEXT provided above.
            2.  If the answer to the user's question cannot be found within the CONTEXT, respond with: "This is beyond my knowledge."
            3. Be concise and helpful.
            4. Do not make up information.
            5. If you cannot find the answer *directly* within the provided context, or if the question is unrelated to the context, you MUST respond with: "This is beyond my knowledge."
            6. You can greet user with "Hello! How can I help you?"


            USER QUESTION: {question}
            """
        )
        response = self.model.generate_content(prompt)
        return response.text.strip()
