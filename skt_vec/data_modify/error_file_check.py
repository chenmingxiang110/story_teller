import os

path = "../../data/txt"
modifie_txt_path = "../../data/modified_txt/"
error_files = ['2491.txt', '1408.txt', '941.txt', '2314.txt', '261.txt', '2039.txt']

print "-------------------------------------"
print "-------------------------------------"
print "-------------------------------------"
for filename in error_files:
    curr_path = modifie_txt_path+filename
    with open(curr_path, 'r') as f:
        i = 0
        for line in f:
            print repr(line)
            print "-------"
            if i==5:
                break
            i += 1
        print "-------------------------------------"
        print "-------------------------------------"
        print "-------------------------------------"
