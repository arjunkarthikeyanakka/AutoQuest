import google.generativeai as genai
import json
import os
import re

# Configure your own gemini-flash-2.0 API key securely
genai.configure(api_key="")

model = genai.GenerativeModel('gemini-2.0-flash')

def generate_scientific_topics(num_topics=100):
    """Generates a list of scientific topics."""
    prompt = f"Generate a list of {num_topics} scientific topics that have published research papers."
    
    try:
        response = model.generate_content(prompt)
        if not response or not hasattr(response, 'text'):
            print("Error: Empty response from model while generating topics.")
            return []
        
        topics = response.text.split("\n")
        topics = [topic.lstrip("0123456789. -") for topic in topics if topic.strip()]
        return topics
    except Exception as e:
        print(f"Error generating topics: {e}")
        return []

def generate_topic_data(topic):
    """Generates scientific content related to a given topic."""
    prompt = (
        f"You are a data generator. I give you a topic, and you generate data accordingly. "
        f"For every topic I give you, create 10 to 14 paragraphs, each approximately 600 characters long. "
        f"The topic is {topic}."
    )
    
    try:
        response = model.generate_content(prompt)
        if not response or not hasattr(response, 'text'):
            print(f"Error: Empty response from model for topic '{topic}'.")
            return None
        return response.text
    except Exception as e:
        print(f"Error generating data for topic '{topic}': {e}")
        return None

def generate_qa_pairs(data):
    """Generates question-answer pairs based on the given text data and ensures JSON format correctness."""
    prompt = (
        f"You are an advanced question generator. Generate 10 to 12 questions per paragraph from the given scientific text. "
        f"At least 80% of the questions should be directly related to the paragraph. Return the output strictly as JSON format "
        f"with the structure of an object: 'question': question, 'answer': answer, 'context': paragraph. These objects must be in an array."
        f"The data is:\n{data}"
    )

    try:
        response = model.generate_content(prompt)
        if not response or not hasattr(response, 'text'):
            print("Error: Empty response from model while generating QA pairs.")
            return None

        qa_json = json.loads(response.text.strip())

        if isinstance(qa_json, list) and all(
            isinstance(item, dict) and {'question', 'answer', 'context'} <= item.keys()
            for item in qa_json
        ):
            return qa_json
        else:
            print("Warning: Incorrect JSON format received. Logging and attempting to correct...")
            log_json_data("invalid_json_log.json", response.text)
            return correct_json_format(response.text)

    except json.JSONDecodeError:
        print("Error: Model response is not valid JSON. Logging and attempting to correct...")
        log_json_data("invalid_json_log.json", response.text)
        return correct_json_format(response.text)
    except Exception as e:
        print(f"Unexpected error generating QA pairs: {e}")
        return None

def correct_json_format(text):
    """Attempts to correct the JSON format if the model response is incorrect."""
    try:
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            extracted_json = match.group(0)
            qa_json = json.loads(extracted_json)

            if isinstance(qa_json, list) and all(
                isinstance(item, dict) and {'question', 'answer', 'context'} <= item.keys()
                for item in qa_json
            ):
                return qa_json

        lines = text.strip().split("\n")
        structured_data = []

        for line in lines:
            parts = line.split(" - ")
            if len(parts) >= 3:
                structured_data.append({
                    "question": parts[0].strip(),
                    "answer": parts[1].strip(),
                    "context": parts[2].strip()
                })

        if structured_data:
            return structured_data

    except json.JSONDecodeError:
        print("Error: Unable to fix JSON format.")
    except Exception as e:
        print(f"Unexpected error in JSON correction: {e}")

    return None

def log_json_data(filename, json_data):
    """Logs the raw JSON output to a file for debugging."""
    try:
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(json.dumps({"raw_response": json_data}, indent=4) + "\n\n")
        print(f"Logged incorrect JSON format to '{filename}'.")
    except Exception as e:
        print(f"Error logging JSON data: {e}")

def append_to_json_file(filename, new_data):
    """Appends new data to an existing JSON file, handling errors gracefully."""
    try:
        existing_data = []
        
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    print(f"Warning: Corrupt JSON file '{filename}', starting fresh.")

        existing_data.extend(new_data)

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=4)

    except Exception as e:
        print(f"Error writing to file '{filename}': {e}")

def main():
    """Main function to orchestrate topic generation, content creation, and QA pair extraction."""
    topics = generate_scientific_topics()
    if not topics:
        print("No topics were generated. Exiting...")
        return

    for topic in topics:
        try:
            print(f"Generating data for: {topic}")
            topic_data = generate_topic_data(topic)

            if not topic_data:
                print(f"Skipping topic '{topic}' due to missing data.")
                continue

            qa_pairs = generate_qa_pairs(topic_data)

            if qa_pairs:
                append_to_json_file("scientific_qa_data.json", qa_pairs)
                print(f"Appended data for topic '{topic}' to 'scientific_qa_data.json'.")
            else:
                print(f"Failed to generate valid QA pairs for topic '{topic}'.")
        except Exception as e:
            print(f"Unexpected error processing topic '{topic}': {e}")

if __name__ == "__main__":
    main()
