import os
files = [f for f in os.listdir('.') if os.path.isfile(f)]
for f in files:
    if f == "transitive_verbs.txt":
        with open(f,"r") as g:
            lines = g.readlines()
            lines2 = [line.strip().split()[0] + "2 " + line.strip().split()[1] + "2 " +line.strip().split()[2] + "2\n" for line in lines]
            lines3 = [line.strip().split()[0] + "3 " + line.strip().split()[1] + "3 " +line.strip().split()[2] + "3\n" for line in lines]
        with open(f,"w") as g2:
            g2.writelines(lines+lines2+lines3)
