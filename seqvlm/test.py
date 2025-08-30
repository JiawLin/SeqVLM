from seqvlm.api import invoke_api



if __name__ == '__main__':
    
    s = """
    '{
    "process": "Image 0 shows a shelf - like structure above a sink and washing machines but no ventilation duct as described. Image 1 and Image 
    3 are similar, showing a ventilation duct above a washing machine. However, there is no clear cabinet as described close to the entrance. Among
    them, Image 3 has a better view of the ventilation duct and the washing machine below, and although the cabinet part is not very distinct as p
    er the description, it is the most suitable among the three images considering the presence of the washing machine and the ventilation duct.",
    "image_id": 3
    } '
    """
    
    
    
    msg = [{
        "role": "assistant",
        "content": s
    }]
    
    
    result = invoke_api('doubao-vision', msg)
    print(result)
    