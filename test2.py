import torch

# a= torch.randn((4,5,512))
# b = a.reshape(4, 5*512)
# print(a)

# b[b>0] =1
# print(b)
# a = torch.ones((5,2,512,1))
# b = torch.zeros((5,2,512,1))
# c = torch.max(torch.cat((a,b), dim=-1),dim=-1)
# print(c[0])

import platform
print('系统:',platform.system())

import time
T1 = time.perf_counter()

#______假设下面是程序部分______
for i in range(100*100):
    pass

T2 =time.perf_counter()
print('程序运行时间:%s毫秒' % ((T2 - T1)*1000))
# t1 = torch.ones(6)
# t1[1:3] = torch.zeros(2)
# t2 = torch.ones(6)
# t3 = torch.div(torch.add(t1,t2),0.5)
# print(t3)
# s = set([(1,2),(2,3),(1,2)])
# s.update(s,[(1,2),(2,5)])
# print((1,2)[1])
# print(list(s))

# t1 = torch.ones(6,device='cuda')
# t2 = torch.ones(6,device='cuda')
# l = [t1,t2]
# print(torch.stack(l).device)
d = {'true':['word','a'], 'pred':['word']}
a = torch.tensor([[1,2,3],[1,2,3],[1,2,3]])
b = torch.tensor([[1],[2],[3]])
m = torch.nn.Softmax()
input = torch.randn(2, 3)
output = m(input)
print(output)
# print(torch.mul(a,b))