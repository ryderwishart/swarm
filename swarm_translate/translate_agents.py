from swarm import Swarm, Agent
from typing import List, Dict
from openai import OpenAI

# Configure OpenAI clients for different models
decision_client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="not-needed"
)

translation_client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="not-needed"
)

# Create Swarm instance with the lightweight decision model
swarm_client = Swarm(client=decision_client)

def transfer_to_translator():
    return translator_bot

def transfer_to_qa():
    return qa_bot

# Define the three specialized agents, using the powerful model for actual work
linguist_bot = Agent(
    name="Linguist",
    instructions="""You are an expert linguist who excels at breaking down sentences into their key components.
    Analyze the given sentence and identify:
    1. Main phrases
    2. Key terms
    3. Idiomatic expressions
    4. Cultural references
    Provide a detailed breakdown that will help with accurate translation.""",
    functions=[transfer_to_translator],
    client=translation_client,  # Use the powerful model for linguistic analysis
    model="llama-3.3-70b-instruct"  # Specify model at the agent level
)

translator_bot = Agent(
    name="Translator",
    instructions="""You are a precise translation expert.
    Your task is to translate each component provided by the linguist:
    1. Maintain the exact meaning of each phrase
    2. Consider multiple possible translations
    3. Note any ambiguities and challenges
    Translate each component separately while maintaining context.""",
    functions=[transfer_to_qa],
    client=translation_client,  # Use the powerful model for translation
    model="llama-3.3-70b-instruct"  # Specify model at the agent level
)

qa_bot = Agent(
    name="QA Expert",
    instructions="""You are a translation quality assurance expert.
    Your role is to:
    1. Research similar sentences and their translations
    2. Evaluate the naturalness of the translation
    3. Suggest improvements for flow and cultural accuracy
    4. Ensure the final translation maintains the original meaning while sounding natural.""",
    client=translation_client,  # Use the powerful model for QA
    model="llama-3.3-70b-instruct"  # Specify model at the agent level
)

def translate_with_agents(text: str) -> str:
    """Main function to coordinate the translation process between agents."""
    
    # Step 1: Linguist bot analyzes the sentence
    response = swarm_client.run(
        agent=linguist_bot,
        messages=[{
            "role": "user",
            "content": f"Please analyze this sentence: '{text}'"
        }]
    )
    analysis = response.messages[-1]["content"]
    
    # Step 2: Translator bot translates components
    response = swarm_client.run(
        agent=translator_bot,
        messages=[{
            "role": "user",
            "content": f"Please translate these components: {analysis}"
        }]
    )
    translations = response.messages[-1]["content"]
    
    # Step 3: QA bot improves the translation
    response = swarm_client.run(
        agent=qa_bot,
        messages=[{
            "role": "user",
            "content": f"""Please improve this translation:
            Original: {text}
            Component translations: {translations}
            Please make it sound natural while maintaining accuracy."""
        }]
    )
    
    return response.messages[-1]["content"]

# Example usage
if __name__ == "__main__":
    test_sentence = "The early bird catches the worm."
    result = translate_with_agents(test_sentence)
    print(f"Original: {test_sentence}")
    print(f"Final translation: {result}") 