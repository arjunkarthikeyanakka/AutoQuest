import google.generativeai as genai
import json
import os
import re
import time
import random

# Configure your own Gemini-flash-2.0 API key securely
genai.configure(api_key="")

model = genai.GenerativeModel('gemini-2.0-flash')

"""
Generates a list of scientific topics. These topics generated are used to get the context-question-answer triplets 
generated.
"""
def generate_scientific_topics(num_topics=100):

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
        handle_rate_limit_error(e)
        return []

"""
Handles rate limit errors by extracting retry delay information. The gemini API's have a set rate limit which is
being handled here. Based on the extraction of the rate limit from the error we set the retry time accordingly. 
"""
def handle_rate_limit_error(error):

    error_str = str(error)
    retry_match = re.search(r'retry_delay\s*{\s*seconds:\s*(\d+)', error_str)

    if retry_match:
        retry_seconds = int(retry_match.group(1))
        print(f"Rate limit exceeded. Waiting for {retry_seconds} seconds before retrying...")
        time.sleep(retry_seconds + 5)
    else:
        print("Rate limit error detected. Waiting for 60 seconds before continuing...")
        time.sleep(60)


"""Generates scientific content related to a given topic."""
def generate_topic_data(topic):
    add_random_delay(3, 5)

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
        handle_rate_limit_error(e)
        return None

"""Adds a random delay to avoid hitting rate limits."""
def add_random_delay(min_seconds=1, max_seconds=3):

    delay = random.uniform(min_seconds, max_seconds)
    print(f"Adding delay of {delay:.2f} seconds...")
    time.sleep(delay)

"""Generates question-answer pairs based on the given text data with difficulty tags."""
def generate_qa_pairs(data):

    paragraphs = data.split('\n\n')
    context_qa_pairs = {}

    for i, paragraph in enumerate(paragraphs):
        if not paragraph.strip():
            continue

        print(f"Processing paragraph {i + 1}/{len(paragraphs)}")

        add_random_delay(2, 5)
        prompt = (
            f"You are a JSON generator for question-answer pairs.\n\n"
            f"TASK: Generate EXACTLY 3 question-answer pairs from the given paragraph.\n\n"
            f"RULES:\n"
            f"1. Each question must have a corresponding answer.\n"
            f"2. Each question must have a difficulty level (easy, medium, or hard). "
            f"3. The question pairs generated should have unequal number of difficulty questions. \n"
            f"4. Return ONLY valid JSON with the following structure.\n"
            f"5. No explanations or text before or after the JSON.\n\n"
            f"JSON STRUCTURE:\n"
            f"{{\n"
            f"  \"context\": \"The paragraph text\",\n"
            f"  \"qa_pairs\": [\n"
            f"    {{\n"
            f"      \"question\": \"What is X?\",\n"
            f"      \"answer\": \"X is Y\",\n"
            f"      \"difficulty\": \"easy\"\n"
            f"    }},\n"
            f"    {{\n"
            f"      \"question\": \"How does Z work?\",\n"
            f"      \"answer\": \"Z works by...\",\n"
            f"      \"difficulty\": \"medium\"\n"
            f"    }},\n"
            f"    {{\n"
            f"      \"question\": \"Why is W important?\",\n"
            f"      \"answer\": \"W is important because...\",\n"
            f"      \"difficulty\": \"hard\"\n"
            f"    }}\n"
            f"  ]\n"
            f"}}\n\n"
            f"PARAGRAPH TO USE:\n{paragraph}\n\n"
            f"RESPONSE (ONLY VALID JSON):"
        )

        try:
            response = model.generate_content(prompt)
            if not response or not hasattr(response, 'text'):
                print(f"Error: Empty response for paragraph {i + 1}.")
                continue

            clean_text = extract_json(response.text)

            try:
                qa_json = json.loads(clean_text)

                if (isinstance(qa_json, dict) and
                        "context" in qa_json and
                        "qa_pairs" in qa_json and
                        isinstance(qa_json["qa_pairs"], list) and
                        len(qa_json["qa_pairs"]) >= 3 and
                        all(isinstance(item, dict) and
                            all(k in item for k in ['question', 'answer', 'difficulty'])
                            for item in qa_json["qa_pairs"])):

                    context = qa_json["context"]
                    if context not in context_qa_pairs:
                        context_qa_pairs[context] = []


                    context_qa_pairs[context].extend(qa_json["qa_pairs"])
                    print(f"Successfully processed paragraph {i + 1}: {len(qa_json['qa_pairs'])} QA pairs added.")
                else:
                    print(f"Warning: JSON format for paragraph {i + 1} is incorrect. Attempting second try.")
                    add_random_delay(2, 4)
                    second_try = generate_qa_with_template(paragraph)
                    if second_try:
                        context = paragraph
                        if context not in context_qa_pairs:
                            context_qa_pairs[context] = []
                        context_qa_pairs[context].extend(second_try)
                        print(f"Second try successful: {len(second_try)} QA pairs added.")
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON in paragraph {i + 1}. Attempting second try.")
                add_random_delay(2, 4)
                second_try = generate_qa_with_template(paragraph)
                if second_try:
                    context = paragraph
                    if context not in context_qa_pairs:
                        context_qa_pairs[context] = []
                    context_qa_pairs[context].extend(second_try)
                    print(f"Second try successful: {len(second_try)} QA pairs added.")

        except Exception as e:
            print(f"Error processing paragraph {i + 1}: {e}")
            handle_rate_limit_error(e)

    final_data = []
    for context, qa_pairs in context_qa_pairs.items():
        final_data.append({
            "context": context,
            "qa_pairs": qa_pairs
        })

    print(f"Total contexts processed: {len(final_data)}")
    return final_data

"""Extracts JSON content from the model response."""
def extract_json(text):
    clean_text = re.sub(r'```json|```', '', text)
    clean_text = clean_text.strip()

    json_pattern = r'\{.*\}'
    match = re.search(json_pattern, clean_text, re.DOTALL)

    if match:
        return match.group(0)

    return clean_text

"""Generates QA pairs using a more structured template approach for difficult cases."""
def generate_qa_with_template(paragraph):

    add_random_delay(4, 6)

    structured_prompt = (
        f"Generate EXACTLY 3 question-answer pairs from this paragraph:\n\n"
        f"{paragraph}\n\n"
        f"Follow this EXACT format for each QA pair (one per line):\n"
        f"{{\"question\": \"What is X?\", \"answer\": \"X is Y\", \"difficulty\": \"easy\"}}\n"
        f"{{\"question\": \"How does Z work?\", \"answer\": \"Z works by...\", \"difficulty\": \"medium\"}}\n"
        f"{{\"question\": \"Why is W important?\", \"answer\": \"W is important because...\", \"difficulty\": \"hard\"}}\n\n"
        f"IMPORTANT: Return ONLY the 3 JSON objects, nothing else."
    )

    try:
        response = model.generate_content(structured_prompt)
        if not response or not hasattr(response, 'text'):
            return None
        result = []
        lines = response.text.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            match = re.search(r'{.*}', line)
            if match:
                try:
                    obj = json.loads(match.group(0))
                    if all(k in obj for k in ['question', 'answer', 'difficulty']):
                        result.append(obj)
                except:
                    continue
        if len(result) < 3:
            try:
                array_match = re.search(r'\[.*\]', response.text, re.DOTALL)
                if array_match:
                    array_json = json.loads(array_match.group(0))
                    for item in array_json:
                        if isinstance(item, dict) and all(k in item for k in ['question', 'answer', 'difficulty']):
                            result.append(item)
            except:
                pass

        return result if result else None
    except Exception as e:
        print(f"Error in template approach: {e}")
        handle_rate_limit_error(e)
        return None

"""Logs the raw JSON output to a file for debugging."""
def log_json_data(filename, json_data):

    try:
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(json.dumps({"raw_response": json_data}, indent=4) + "\n\n")
        print(f"Logged incorrect JSON format to '{filename}'.")
    except Exception as e:
        print(f"Error logging JSON data: {e}")

"""Appends new data to an existing JSON file, handling errors gracefully."""
def append_to_json_file(filename, new_data):

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

"""Validates and filters contexts to ensure each has the minimum required questions."""
def validate_min_questions_per_context(context_data, min_questions=3):

    if not context_data:
        return []


    valid_contexts = []
    for context_item in context_data:
        if len(context_item["qa_pairs"]) >= min_questions:
            valid_contexts.append(context_item)
        else:
            print(
                f"Excluding context with only {len(context_item['qa_pairs'])} questions (minimum {min_questions} required)")

    return valid_contexts

"""Process a topic with retry logic for rate limit errors."""
def process_topic_with_retry(topic, max_retries=3):

    for attempt in range(1, max_retries + 1):
        try:
            print(f"Generating data for: {topic} (Attempt {attempt}/{max_retries})")
            topic_data = generate_topic_data(topic)

            if not topic_data:
                print(f"Skipping topic '{topic}' due to missing data.")
                return False

            qa_context_data = generate_qa_pairs(topic_data)

            if qa_context_data:
                validated_contexts = validate_min_questions_per_context(qa_context_data, min_questions=3)

                if validated_contexts:
                    add_random_delay(1, 2)
                    append_to_json_file("scientific_qa_data_structured.json", validated_contexts)
                    total_qa_pairs = sum(len(ctx["qa_pairs"]) for ctx in validated_contexts)
                    print(
                        f"Appended {len(validated_contexts)} contexts with {total_qa_pairs} QA pairs for topic '{topic}' to 'scientific_qa_data_structured.json'.")
                    return True
                else:
                    print(f"No contexts with minimum required questions for topic '{topic}'.")
                    return False
            else:
                print(f"Failed to generate valid QA pairs for topic '{topic}'.")
                return False

        except Exception as e:
            print(f"Error processing topic '{topic}' (Attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                backoff_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"Retrying in {backoff_time:.2f} seconds...")
                time.sleep(backoff_time)
            else:
                print(f"Maximum retries reached for topic '{topic}'. Moving to next topic.")
                return False

    return False

"""Main function to orchestrate topic generation, content creation, and QA pair extraction."""
def main():

    try:
        if os.path.exists("cached_topics.json"):
            with open("cached_topics.json", "r") as f:
                topics = json.load(f)
            print(f"Loaded {len(topics)} topics from cache.")
        else:
            print("Waiting 3 seconds before starting...")
            time.sleep(3)
            topics = generate_scientific_topics()

            if topics:
                with open("cached_topics.json", "w") as f:
                    json.dump(topics, f)
                print(f"Cached {len(topics)} topics for future use.")

        if not topics:
            print("No topics were generated. Exiting...")
            return

        for i, topic in enumerate(topics):
            print(f"\nProcessing topic {i + 1}/{len(topics)}: {topic}")

            success = process_topic_with_retry(topic)

            if i < len(topics) - 1:
                delay = 7
                print(f"Topic completed. Waiting {delay:.2f} seconds before next topic...")
                time.sleep(delay)

    except KeyboardInterrupt:
        print("\nScript interrupted by user. Saving progress...")
    except Exception as e:
        print(f"Unexpected error in main function: {e}")


if __name__ == "__main__":
    main()