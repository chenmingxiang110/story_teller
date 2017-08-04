import os
import re

punc_alpha_list = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',',','.',' ','?','!']
alpha_list = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',',']

path = "../../data/txt"
modifie_txt_path = "../../data/modified_txt/"
sent_path = "../../data/sentences/"

num_list = []
sentend = [".","?","!"]
i = 1
for filename in os.listdir(modifie_txt_path):
    if i%10 == 0:
        print i
    i+= 1
    curr_path = modifie_txt_path+filename
    new_path = sent_path+filename
    sent_list = []
    with open(curr_path, 'r') as f:
        article = f.readline()
        article = article.replace("Mr. ", "Mr.")
        article = article.replace("Ms. ", "Ms.")
        article = article.replace("Mrs. ", "Mrs.")
        article = article.replace("Miss. ", "Miss.")
        article = article.replace("Prof. ", "Prof.")
        article = article.replace("Professor. ", "Professor.")
        article = article.replace("Dr. ", "Dr.")
        out = "".join(c for c in article if c.lower() in punc_alpha_list)
        out = ' '.join(out.split())
        sent_list = re.split('\. |\? |\! ', out)
    with open(new_path, 'w+') as f:
        for item in sent_list:
            if len(item)>40:
                f.write(item)
                f.write("\n")
