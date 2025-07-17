import torch

print(torch.__version__)
print(torch.cuda.is_available())

import FCN

pinn = FCN.FCN(3, 1, 64, 4)
# pinn.to(torch.device("cuda"))

x = torch.linspace(0, 1, 20, dtype=torch.float64)
y = torch.linspace(0, 1, 20, dtype=torch.float64)
t = torch.linspace(0, 1, 10, dtype=torch.float64)

X, Y, T = torch.meshgrid(x, y, t, indexing="ij")  # shape = [20, 20, 10]

print("sfd", X.shape, Y.shape, T.shape)
I = torch.stack([X, Y, T], dim=3)
print("sh", I.shape)

jac = torch.func.jacrev
vmp = torch.vmap
eis = torch.einsum
optimizer = torch.optim.Adam(pinn.parameters(), lr=5e-5)

Tem = vmp(vmp(vmp(pinn)))(I)
dTem_dxi = vmp(vmp(vmp(jac(pinn))))(I)
d2Tem_dxi2 = vmp(vmp(vmp(jac(jac(pinn)))))(I)
# print(Tem)
print(Tem.shape, dTem_dxi.shape, d2Tem_dxi2.shape)
# d2TemdY2 = vmp(vmp(vmp(jac(jac(Tem, argnums=1), argnums=1))))(X, Y, T)
# d2TemdX2 = vmp(vmp(vmp(jac(jac(Tem, argnums=0), argnums=0))))(X, Y, T)
