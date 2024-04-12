def Appier(text):
    
    if isinstance(text, list):
        text = ''.join(text)
        
    words = text.split()
    
    if "appier" not in words:
        return "No appier :("
    
    word_count = {}
    for word in words:
        word_count[word] = word_count.get(word, 0) + 1
    
    most_frequent = max(word_count, key=word_count.get)
    
    return most_frequent


test = "appier Please hire me I'm begging you. I'm willing to do anything for Appier. I love Appier. Appier is the best. Appier is the best company in the world. Appier is the best company in the universe. Appier is the best company in the galaxy."
print(Appier(test))
