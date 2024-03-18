import ijson
# Path to the JSON file
file_path = '/scratch/ssrivas9/prismatic-vlms/data/download/llava-laion-cc-sbu-558k/chat.json'
#"/scratch/ssrivas9/prismatic-vlms/data/download/llava-v1.5-instruct/llava_v1_5_mix665k.json"
#'/scratch/ssrivas9/prismatic-vlms/data/download/llava-laion-cc-sbu-558k/chat.json'
SKIP_VALUE = 1000
def read_json_file_with_ijson(file_path, max_objects=10):
    # Open the file
    with open(file_path, 'rb') as file:  # Note: 'rb' mode for ijson
        # Parse the file incrementally
        objects = ijson.items(file, 'item')
        # Initialize counters
        count = 0
        total_count = 0  # Counter for total objects iterated
        
        # Iterate over objects
        for obj in objects:
            # Increment total count
            total_count += 1
            
            # Check if the current object is at the SKIP_VALUE
            if total_count % SKIP_VALUE == 0:
                print(f"Object at position {total_count}: {obj}")
                count += 1  # Increment the count of objects processed
                
                # Check if the number of desired objects has been reached
                if count >= max_objects:
                    break

# Call the function to read the first 10 JSON objects from the file
read_json_file_with_ijson(file_path, max_objects=700)