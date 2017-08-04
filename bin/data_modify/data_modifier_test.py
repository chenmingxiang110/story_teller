import os

path = "../../data/txt"
modifie_txt_path = "../../data/modified_txt/"

num_list = []
i = 1
for filename in os.listdir(modifie_txt_path):
    if i%10 == 0:
        print i
    i+= 1
    curr_path = modifie_txt_path+"/"+filename
    with open(curr_path, 'r') as f:
        num = 0
        for line in f:
            num += 1
        if num != 1:
            num_list.append(filename)
print num_list
