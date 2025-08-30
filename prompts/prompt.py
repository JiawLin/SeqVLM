SYSTEM_PROMPT = """Imagine you are in a room and you are aksed to find one object.

Given a series of images and a query describing a specific object in the room, you need to analyze the images, and find an image that best fits the query.

Please note that each image is composed of sub-images displaying the object from multiple perspectives. In each sub-image, there is a red rectangle box highlighting the object, but the box may also contain other irrelevant objects. You need to make a selection by combining the object in the red rectangle box with surrounding environment from different perspective images.

Return the index of the image where the object is found, and describe the process of selecting this image.

Your response should be in the following format, and it should not include code block markers such as ```json.

{
  "process": "Explain the process of how you identified the room's features and located the target object",
  "image_id": 1 # Replace with the actual index based on the input order of images, starting from 0. 
}

Here is an example for you.

```
Input: 
Query: Find the black table that is surrounded by four chairs.
Here are the images of 3 possible objects.
[image_0, image_1, image_2]

Output:
{
  "process": "After carefully examining all the input images, I found only the tables in image_1, image_2 are black, but only the tables in image_2 has is surrounded by four chairs. So the correct object is the table in image_2",
  "image_id": 2
}

```

Here are some tips:
# Please follow the format of the example strictly
# If there is no object that fully matches the query, select the most suitable one. 
# If the types of all objects are inconsistent with the query, output -1 in the value of image_id.

"""

USER_PROMPT = """Query: {query}
Here are the images of {n_images} possible objects."""

IMAGE_ID_INVALID_PROMPT = """The image_id {image_id} you selected does not exist. Did you perhaps see it incorrectly? Please reconsider and select another image. Remember to reply using JSON format with the two keys "process", "image_id" as required before."""

WRONG_FORMAT_PROMPT = """The answer contains extra characters. Please follow the format of the example strictly."""