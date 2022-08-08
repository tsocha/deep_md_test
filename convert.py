import pickle
import numpy as np

root = "data_convert"
contents = pickle.load(open(f"{root}/data.pkl", "rb"))

# We only need the following inputs for inference
input_names = ['coord', 'type', 'natoms_vec', 'box', 'default_mesh']
for input_name in input_names:
  assert input_name in contents.keys()
  name = input_name
  value = contents[name]
  value = np.squeeze(value, axis=None)
  print(name, value.dtype)
  value = value.astype('float32')

  # Write to binary with format:
  # rank, shape, data
  shape = value.shape
  rank = len(shape)
  tensor = [None for i in range(1 + rank + value.size)]
  
  tensor[0] = rank
  tensor[1:rank+1] = shape
  tensor[rank+1:] = value.flatten()
  tensor = np.array(tensor).astype('float32')
  
  filepath = f"{root}/{input_name}.bin"
  tensor.tofile(filepath)

