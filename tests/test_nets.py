import easynets as en

def test_create_network(
  num_layers=5,
  layers=[2, 5, 5, 1], # input and output included
  activation=['relu', 'relu', 'relu', 'sigmoid'])
