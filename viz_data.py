import ijson
# Path to the JSON file
#"/scratch/ssrivas9/prismatic-vlms/data/download/llava-v1.5-instruct/llava_v1_5_mix665k.json"
#'/scratch/ssrivas9/prismatic-vlms/data/download/llava-laion-cc-sbu-558k/chat.json'
SKIP_VALUE = 1000
def read_json_file_with_ijson(file_path, max_objects=None):
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
                if max_objects is not None:
                    if count >= max_objects:
                        break

# Call the function to read the first 10 JSON objects from the file
pretrain_data = '/localdisk/ssrivas9/prismatic-vlms/data/download/llava-laion-cc-sbu-558k/chat.json'
instruct_data = '/localdisk/ssrivas9/prismatic-vlms/data/download/llava-v1.5-instruct/llava_v1_5_mix665k.json'
vqd_data = '/localdisk/ssrivas9/datasets/VQD_dataset/train.json'
instruct_158k_data = '/localdisk/ssrivas9/prismatic-vlms/data/continual/llava-158k-instruct/llava_instruct_150k-cleaned.json'
instruct_80k_data = '/localdisk/ssrivas9/prismatic-vlms/data/continual/llava-80k-instruct/llava_instruct_80k.json'
read_json_file_with_ijson(instruct_80k_data, max_objects=None)