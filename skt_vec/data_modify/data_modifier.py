import os

path = "../../data/txt"
modifie_txt_path = "../../data/modified_txt/"

i = 1
for filename in os.listdir(path):
    if i%10 == 0:
        print i

    curr_path = path+"/"+filename
    newfilename = repr(i)+".txt"
    i += 1

    with open(curr_path, 'r') as f:
        line_list = []
        for line in f:
            real_line = line.split("\n")[0]
            real_line = line.split("\r")[0]
            if len(real_line) == 0:
                continue
            line_list.append(real_line)
        with open(modifie_txt_path + newfilename, 'w+') as fw:
            for aline in line_list:
                fw.write(aline)
                fw.write(" ")
