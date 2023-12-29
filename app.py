from word_processor import PhraseSimilarity

#User need to enter the phrase inorder to get the similarilty phrase
input_phrase = input("Enter your phrase here:")
# Task 1: Downloading and saving the Word2Vec vectors
# Replace 'location' with the path to the downloaded pretrained vectors
location = r'D:\GoogleNews-vectors-negative300.bin.gz'  # Update with your file path

# Initialize PhraseSimilarity class
phrase_sim = PhraseSimilarity(location, r'D:\phrases.csv')

# Task 2: Load the processed word embeddings and phrases.csv
similarities_df = phrase_sim.compute_similarities()
# Example of finding closest match to a given string

closest_phrase, similarity = phrase_sim.closest_match(input_phrase)
print(f"Closest phrase: {closest_phrase}, Similarity: {similarity}")
