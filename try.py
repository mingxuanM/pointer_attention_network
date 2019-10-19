import torch

encoder_context = torch.arange(40).view(10,4).float()
print(type(encoder_context))
print(encoder_context)

output_context = torch.randn(4)
output_context = output_context.repeat(encoder_context.size(0),1)
output_context = output_context
print(type(output_context))
print(output_context)

full_context = torch.cat([encoder_context,output_context], dim=1)
print(full_context)