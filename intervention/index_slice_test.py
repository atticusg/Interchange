import torch, re

class MyList:
    def __init__(self, l):
        self.l = l

    def get(self, *idx):
        print(*idx)

    def __getitem__(self, item):
        print(item)
        # return self.l[item]

a = MyList([[0,1,2,3,4],[10,11,12,13,14],[20,21,22,23,24]])

def str_to_slice(s):
    return slice(*map(lambda x: int(x.strip()) if x.strip() else None, s.split(':')))

print(str_to_slice("::"))

a = torch.tensor([1,2,3])
print(a[True, True, True])
s = "  ... "
print(s.strip())


s = "activation[1:5]"
s2 = "activation"
res = re.search("\[.*\]", s)
idx = res.group().strip("[]")
print(idx)

res = re.search("\[.*\]", s2)
print(res)